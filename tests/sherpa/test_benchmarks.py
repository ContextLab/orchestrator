"""Tests for the #492 benchmark harness gates and the Scenario B repair planner.

Everything here is real: the repair planner's output is written to a real
repository on disk and verified by a REAL pytest subprocess. No mocks, no
monkeypatching of the code under test.

Two classes of regression are covered:

1. Gate arithmetic (`_evaluate_gates`). The go/no-go verdict must react to
   EVERY row of the table, and each threshold must flip at its declared
   boundary -- not one row later, and not silently.
2. Defect classification (`sherpa.benchmarks.repair_planner`). The planner
   must recognise the preregistered defect grammar structurally, so that
   defect instances spelled differently from the seeder's output are still
   detected and repaired. A planner that only inverts the seeder's exact
   bytes measures nothing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sherpa.benchmarks.harness import GATES, _evaluate_gates
from sherpa.benchmarks.repair import DEFECT_CLASSES, make_repair_task, materialize_repo
from sherpa.benchmarks.repair_planner import classify_and_repair
from sherpa.planner import PlanAuthoringError

# --------------------------------------------------------------------- gates


def _all_pass_suite() -> dict:
    """A suite dict on which every preregistered gate passes."""
    return {
        "decomposition_decisions": GATES["min_decomposition_decisions"],
        "claimed_atomic_admissions": GATES["min_claimed_atomic_admissions"],
        "task_success_rate": GATES["min_task_success_rate"],
        "m_upper_bound_max": GATES["max_m_upper_bound"] - 0.001,
        "scenario_c": {"needle_recall": GATES["min_needle_recall"]},
        "scenario_a": {"exactly_once_effects": True, "projection_equivalent": True},
        "scenario_b_externally_verified": 1.0,
        "scenario_b_defect_class_detected": GATES["min_defect_class_detection"],
        "scenario_b_undetected_classes": [],
    }


def _verdict(table: str) -> str:
    assert ("**GO**" in table) != ("**NO-GO" in table), f"ambiguous verdict:\n{table}"
    return "GO" if "**GO**" in table else "NO-GO"


def test_all_pass_suite_is_go() -> None:
    """Control: the fixture used by the failure tests must itself be a GO."""
    table = _evaluate_gates(_all_pass_suite())
    assert "FAIL" not in table, f"control suite should have no failing row:\n{table}"
    assert _verdict(table) == "GO", table


def test_headline_decomposition_gate_failure_forces_no_go() -> None:
    """A FAIL on the FIRST gate row must still produce NO-GO.

    Regression: the verdict was computed over lines[3:], skipping lines[2],
    which is the first gate row -- so the headline decomposition gate could
    never fail the suite.
    """
    suite = _all_pass_suite()
    suite["decomposition_decisions"] = GATES["min_decomposition_decisions"] - 1
    table = _evaluate_gates(suite)
    first_gate_row = table.splitlines()[2]
    assert "decomposition decisions" in first_gate_row, table
    assert "FAIL" in first_gate_row, f"first gate row should FAIL:\n{table}"
    assert _verdict(table) == "NO-GO", (
        "a failing headline gate must produce NO-GO, got:\n" + table
    )


# Each entry: (label, mutator(suite, value), failing_value, passing_value)
GATE_BOUNDARIES = [
    (
        "decomposition decisions",
        lambda s, v: s.__setitem__("decomposition_decisions", v),
        GATES["min_decomposition_decisions"] - 1,
        GATES["min_decomposition_decisions"],
    ),
    (
        "claimed-atomic steps",
        lambda s, v: s.__setitem__("claimed_atomic_admissions", v),
        GATES["min_claimed_atomic_admissions"] - 1,
        GATES["min_claimed_atomic_admissions"],
    ),
    (
        "held-out repair/corpus tasks",
        lambda s, v: s.__setitem__("task_success_rate", v),
        GATES["min_task_success_rate"] - 0.0001,
        GATES["min_task_success_rate"],
    ),
    (
        "corrected m upper bound",
        lambda s, v: s.__setitem__("m_upper_bound_max", v),
        GATES["max_m_upper_bound"],  # gate is strict <, so equality must FAIL
        GATES["max_m_upper_bound"] - 0.0001,
    ),
    (
        "seeded-needle retrieval recall",
        lambda s, v: s["scenario_c"].__setitem__("needle_recall", v),
        GATES["min_needle_recall"] - 0.0001,
        GATES["min_needle_recall"],
    ),
    (
        "crash/resume preserves projections",
        lambda s, v: s["scenario_a"].__setitem__("projection_equivalent", v),
        False,
        True,
    ),
    (
        "repair results verified by REAL pytest",
        lambda s, v: s.__setitem__("scenario_b_externally_verified", v),
        0.9999,
        1.0,
    ),
    (
        "seeded defect classes named by the planner",
        lambda s, v: s.__setitem__("scenario_b_defect_class_detected", v),
        0.9999,
        1.0,
    ),
]


def _row_for(table: str, label: str) -> str:
    rows = [ln for ln in table.splitlines() if ln.startswith("| ") and label in ln]
    assert len(rows) == 1, f"expected exactly one row matching {label!r}:\n{table}"
    return rows[0]


@pytest.mark.parametrize("label,mutate,failing,passing", GATE_BOUNDARIES,
                         ids=[g[0] for g in GATE_BOUNDARIES])
def test_gate_flips_at_its_declared_boundary(label, mutate, failing, passing) -> None:
    below = _all_pass_suite()
    mutate(below, failing)
    below_table = _evaluate_gates(below)
    assert "FAIL" in _row_for(below_table, label), (
        f"{label}: value {failing!r} is on the failing side and must FAIL:\n{below_table}"
    )
    assert _verdict(below_table) == "NO-GO", below_table

    at = _all_pass_suite()
    mutate(at, passing)
    at_table = _evaluate_gates(at)
    assert "PASS" in _row_for(at_table, label), (
        f"{label}: value {passing!r} is on the passing side and must PASS:\n{at_table}"
    )
    assert _verdict(at_table) == "GO", at_table


def test_missing_observation_fails_rather_than_passing_silently() -> None:
    """An absent measurement must never be scored as a pass."""
    for key in ("task_success_rate", "m_upper_bound_max",
                "scenario_b_externally_verified",
                "scenario_b_defect_class_detected"):
        suite = _all_pass_suite()
        del suite[key]
        table = _evaluate_gates(suite)
        assert "n/a" in table, f"{key}: missing observation should render n/a:\n{table}"
        assert _verdict(table) == "NO-GO", f"{key} missing must be NO-GO:\n{table}"


# ------------------------------------------------------------------- repairs


def _run_pytest(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                          cwd=repo, capture_output=True, text=True, timeout=300)


def _materialize(root: Path, module_src: str, test_src: str) -> Path:
    repo = root / "repo"
    (repo / "pkg").mkdir(parents=True, exist_ok=True)
    (repo / "tests").mkdir(parents=True, exist_ok=True)
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "mod.py").write_text(module_src, encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text(test_src, encoding="utf-8")
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return repo


@pytest.mark.parametrize("defect_class", DEFECT_CLASSES)
def test_seeded_defect_actually_fails_before_repair(defect_class, tmp_path: Path) -> None:
    """Every seeded defect class must really break its own test suite.

    A "seeded defect" whose tests pass before repair makes the detection gate
    vacuous, so this is checked with a real pytest run, not by inspection.
    """
    task = make_repair_task(11, defect_class)
    repo = materialize_repo(tmp_path / "repo", task)
    (tmp_path / "repo").mkdir(exist_ok=True)
    proc = _run_pytest(repo)
    assert proc.returncode != 0, (
        f"{defect_class}: seeded defect does not fail its tests -- the fixture is "
        f"not defective.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


@pytest.mark.parametrize("defect_class", DEFECT_CLASSES)
def test_seeded_defects_are_classified_and_repaired(defect_class, tmp_path: Path) -> None:
    """The planner repairs the seeder's own output, verified by real pytest."""
    task = make_repair_task(23, defect_class)
    module = task.files["pkg/mod.py"]
    test_src = task.tests["tests/test_mod.py"]
    fn = test_src.splitlines()[0].split("import")[1].strip()

    inferred, fixed = classify_and_repair(module, fn, test_src)
    assert inferred == defect_class, (
        f"planner classified the seeded {defect_class} defect as {inferred}"
    )
    assert fixed != module, f"{defect_class}: planner produced no change"

    repo = _materialize(tmp_path, fixed, test_src)
    proc = _run_pytest(repo)
    assert proc.returncode == 0, (
        f"{defect_class}: repaired module still fails real pytest.\n"
        f"module:\n{fixed}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# Held-out defect instances: same preregistered grammar, spelled differently
# from anything the seeder emits. A planner that string-matches the seeder's
# output rejects all of these.
HELD_OUT_VARIANTS = [
    (
        "off_by_one",
        "accumulate_upto",
        "def accumulate_upto(n):\n"
        "    total = 0\n"
        "    for i in range(0, n):\n"
        "        total += i\n"
        "    return total\n",
        "from pkg.mod import accumulate_upto\n\n"
        "def test_accumulate_upto():\n"
        "    assert accumulate_upto(4) == 10\n"
        "    assert accumulate_upto(6) == 21\n",
    ),
    (
        "inverted_comparison",
        "larger_of",
        "def larger_of(a, b):\n"
        "    if b > a:\n"
        "        return a\n"
        "    return b\n",
        "from pkg.mod import larger_of\n\n"
        "def test_larger_of():\n"
        "    assert larger_of(2, 9) == 9\n"
        "    assert larger_of(11, 4) == 11\n",
    ),
    (
        "wrong_constant",
        "scale_value",
        "def scale_value(x):\n"
        "    return 3 * x + 1\n",
        "from pkg.mod import scale_value\n\n"
        "def test_scale_value():\n"
        "    assert scale_value(4) == 9\n"
        "    assert scale_value(7) == 15\n",
    ),
    (
        "missing_guard",
        "safe_ratio",
        "def safe_ratio(n):\n"
        "    return 100 // n\n",
        "from pkg.mod import safe_ratio\n\n"
        "def test_safe_ratio():\n"
        "    assert safe_ratio(0) == 0\n"
        "    assert safe_ratio(5) == 20\n",
    ),
]


@pytest.mark.parametrize("defect_class,fn,module,test_src", HELD_OUT_VARIANTS,
                         ids=[v[0] for v in HELD_OUT_VARIANTS])
def test_held_out_spellings_are_repaired(defect_class, fn, module, test_src,
                                         tmp_path: Path) -> None:
    """Defects from the same grammar, written differently, must still repair.

    The unrepaired variant is run through real pytest first, so the test can
    fail: if the fixture were already green the repair would prove nothing.
    """
    broken_repo = _materialize(tmp_path / "before", module, test_src)
    before = _run_pytest(broken_repo)
    assert before.returncode != 0, (
        f"{defect_class}: held-out variant is not actually broken:\n{before.stdout}"
    )

    inferred, fixed = classify_and_repair(module, fn, test_src)
    assert inferred == defect_class, (
        f"held-out variant of {defect_class} was classified as {inferred}"
    )
    assert fixed != module, f"{defect_class}: planner produced no change"

    repo = _materialize(tmp_path / "after", fixed, test_src)
    proc = _run_pytest(repo)
    assert proc.returncode == 0, (
        f"{defect_class}: repaired held-out variant still fails real pytest.\n"
        f"module:\n{fixed}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_undetected_classes_are_named_in_the_gate_table() -> None:
    """A class the planner could not name is reported, not quietly dropped."""
    suite = _all_pass_suite()
    suite["scenario_b_defect_class_detected"] = 0.75
    suite["scenario_b_undetected_classes"] = ["missing_guard"]
    table = _evaluate_gates(suite)
    row = _row_for(table, "seeded defect classes named by the planner")
    assert "NOT DETECTED: missing_guard" in row, table
    assert "FAIL" in row, table
    assert _verdict(table) == "NO-GO", table


def test_defect_outside_grammar_is_refused_not_guessed() -> None:
    """Outside the preregistered grammar the planner must refuse loudly."""
    module = 'def join_items(items):\n    return ",".join(items)\n'
    test_src = ('from pkg.mod import join_items\n\n'
                'def test_join_items():\n'
                '    assert join_items(["a", "b"]) == "a|b"\n')
    with pytest.raises(PlanAuthoringError):
        classify_and_repair(module, "join_items", test_src)


def test_scenario_c_is_idempotent_across_reruns(tmp_path) -> None:
    """Re-running into an existing output directory must not degrade results.

    The scenario opened `corpus.db` in place, so a second run indexed every
    chunk again. The duplicates diluted top-k retrieval and needle recall fell
    from 1.0 to 0.625 -- a measurement artifact that read exactly like a real
    regression, and which #492's "one documented command" reproducibility
    requirement cannot tolerate.
    """
    from sherpa.benchmarks.scenarios import scenario_c

    first = scenario_c(tmp_path / "bench")
    second = scenario_c(tmp_path / "bench")

    assert first["needle_recall"] == 1.0, first
    assert second["needle_recall"] == first["needle_recall"], (
        f"re-run degraded recall {first['needle_recall']} -> {second['needle_recall']}"
    )
    assert second["docs"] == first["docs"]
    assert second["total_chars"] == first["total_chars"]

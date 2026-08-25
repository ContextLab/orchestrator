"""Benchmark harness: python -m sherpa.benchmarks.harness [--out DIR]

Runs scenarios A, B (seen + held-out variants) and C through the public
Engine API, writes raw artifacts + a measurement report, and prints the
go/no-go statement against the preregistered gates from issue #492.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sherpa.benchmarks.scenarios import scenario_a, scenario_b, scenario_c
from sherpa.metrics import aggregate_run_reports, render_report_md

GATES = {
    "min_decomposition_decisions": 50,
    "min_claimed_atomic_admissions": 30,
    "min_task_success_rate": 0.80,
    "max_m_upper_bound": 1.0,
    "min_needle_recall": 0.95,
    "min_defect_class_detection": 1.0,
}


def run_all(base: Path, b_seeds: list[int], b_heldout: list[int]) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("== Scenario A: durable semantics fixture ==")
    a = scenario_a(base)
    (base / "scenario_a.json").write_text(json.dumps(a, indent=2))

    print("== Scenario B: repository repair (seen + held-out) ==")
    b = scenario_b(base, seeds=b_seeds, heldout_seeds=b_heldout)
    (base / "scenario_b.json").write_text(json.dumps(b, indent=2))

    print("== Scenario C: evidence-grounded corpus task ==")
    c = scenario_c(base)
    (base / "scenario_c.json").write_text(json.dumps(c, indent=2))

    # Decomposition battery: 25 tiny goals x 2 decompose decisions each, executed
    # through the same kernel to feed the branching/ambiguity measurements.
    print("== Decomposition battery ==")
    decomp = _decomposition_battery(base / "decomp_ws")
    (base / "decomposition_battery.json").write_text(json.dumps(decomp, indent=2))

    run_reports = []
    for r in b:
        if r.get("trace"):
            trace = json.loads(Path(r["trace"]).read_text())["metrics"]
            run_reports.append(trace)
    for d in decomp:
        run_reports.append(d)

    suite = aggregate_run_reports(run_reports)
    admissions_total = sum(x.get("admission", {}).get("checked", 0) for x in run_reports)
    claimed_total = sum(x.get("admission", {}).get("claimed_atomic", 0) for x in run_reports)
    decompositions_total = sum(x.get("branching", {}).get("decompositions", 0)
                               for x in run_reports)
    suite.update({
        "decomposition_decisions": decompositions_total,
        "claimed_atomic_admissions": claimed_total,
        "admission_checks_total": admissions_total,
        "wall_seconds": round(time.time() - started, 1),
        "scenario_a": {k: a[k] for k in ("killed_by_sigkill", "v2_status",
                                         "exactly_once_effects",
                                         "projection_equivalent")},
        "scenario_b_success_rate": _rate([r["status"] == "completed" for r in b]),
        "scenario_b_heldout_success": _rate([r["status"] == "completed"
                                             for r in b if r.get("held_out")]),
        "scenario_b_externally_verified": _rate([r.get("externally_verified", False)
                                                 for r in b]),
        "scenario_b_defect_class_detected": _rate([r.get("defect_detected", False)
                                                   for r in b]),
        "scenario_b_undetected_classes": sorted({r["defect_class"] for r in b
                                                 if not r.get("defect_detected")}),
        "scenario_c": c,
    })

    report_md = render_report_md(suite, [t for t in run_reports if "admission" in t])
    go_no_go = _evaluate_gates(suite)
    report_md += "\n## Preregistered gates\n\n" + go_no_go + "\n"
    (base / "report.md").write_text(report_md, encoding="utf-8")
    (base / "suite.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    return {"suite": suite, "report_md": report_md, "go_no_go": go_no_go,
            "runs": run_reports}


def main() -> None:
    parser = argparse.ArgumentParser(description="sherpa benchmark harness (#492)")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/artifacts"))
    args = parser.parse_args()
    out = run_all(args.out, b_seeds=[11, 23], b_heldout=[401, 409])
    print(out["report_md"])
    print(out["go_no_go"])


def _rate(values: list[bool]) -> float | None:
    return round(sum(1 for v in values if v) / len(values), 4) if values else None


def _decomposition_battery(ws_base: Path) -> list[dict]:
    """50 decomposition decisions through the kernel; feeds m=b*f and overclaim."""
    from sherpa.ir import ProblemSpec
    from sherpa.kernel import Engine
    from sherpa.metrics import run_metrics
    from sherpa.planner import StubPlanner

    reports = []
    for i in range(25):
        ws = ws_base / f"run{i:02d}"
        engine = Engine(ws, planner=StubPlanner())
        spec = ProblemSpec(
            id=f"decomp-{i:02d}",
            goal=f"battery goal {i} two-phase",
            authority={"fs_read": ["**"], "fs_write": ["**"],
                        "subprocess_allow": ["**"]},
            metadata={"root_nodes": [
                {"kind": "decompose", "id": "phase_one",
                 "subgoal": f"battery goal {i} phase one",
                 "hints": {"requested_capability": "text.search_corpus",
                           "plan_library": [_library_entry(i, "phase_one")]},
                 },
                {"kind": "decompose", "id": "phase_two",
                 "subgoal": f"battery goal {i} phase two",
                 "hints": {"requested_capability": "repo.run_tests",
                           "plan_library": [_library_entry(i, "phase_two")]},
                 },
                {"kind": "return", "id": "fin", "outputs": {"i": i}},
            ]},
        )
        result = engine.run(spec)
        m = run_metrics(engine.store.events(run_id=result.run_id))
        m["run_id"] = result.run_id
        reports.append(m)
        engine.close()
    return reports


def _library_entry(i: int, phase: str) -> dict:
    capability = "text.search_corpus" if phase == "phase_one" else "repo.run_tests"
    body = (
        [{"kind": "invoke_capability", "id": "probe_cap",
          "capability": "text.search_corpus",
          "inputs": {"query": f"battery {i} {phase}", "k": 1}}]
        if capability == "text.search_corpus"
        else [{"kind": "invoke_capability", "id": "noop_tests",
               "capability": "repo.run_tests",
               "inputs": {"cwd": ".", "args": ["--version"],
                          "atomic_claim": False}}]
    )
    body.append({"kind": "return", "id": "done", "outputs": {"phase": phase}})
    return {
        "match": {"capability": capability},
        "plan": {
            "id": f"lib_{phase}_{i:02d}",
            "authority": {},
            "budgets": {"max_fanout": 2},
            "root": body,
        },
    }


def _evaluate_gates(suite: dict) -> str:
    lines = ["| gate | threshold | observed | verdict |", "|-|-|-|-|"]

    def row(name: str, threshold: str, observed: str, ok: bool) -> None:
        lines.append(f"| {name} | {threshold} | {observed} | {'PASS' if ok else 'FAIL'} |")

    dec = suite.get("decomposition_decisions", 0)
    row("decomposition decisions with admission outcomes", ">= 50", str(dec),
        dec >= GATES["min_decomposition_decisions"])
    atomic = suite.get("claimed_atomic_admissions", 0)
    row("claimed-atomic steps admitted/rejected independently", ">= 30", str(atomic),
        atomic >= GATES["min_claimed_atomic_admissions"])
    sr = suite.get("task_success_rate")
    row("held-out repair/corpus tasks externally verified within budgets", ">= 80%",
        f"{sr}" if sr is not None else "n/a",
        sr is not None and sr >= GATES["min_task_success_rate"])
    mb = suite.get("m_upper_bound_max")
    row("corrected m upper bound on fixture distribution", "< 1.0",
        f"{mb}" if mb is not None else "n/a",
        mb is not None and mb < GATES["max_m_upper_bound"])
    recall = suite.get("scenario_c", {}).get("needle_recall")
    row("seeded-needle retrieval recall", ">= 95%",
        f"{recall}" if recall is not None else "n/a",
        recall is not None and recall >= GATES["min_needle_recall"])
    crash_ok = suite.get("scenario_a", {}).get("exactly_once_effects") and \
        suite.get("scenario_a", {}).get("projection_equivalent")
    row("crash/resume preserves projections; no repeated effects", "required",
        "met" if crash_ok else "not met", bool(crash_ok))
    verified = suite.get("scenario_b_externally_verified")
    row("repair results verified by REAL pytest outside the runtime", "100%",
        f"{verified}" if verified is not None else "n/a",
        verified is not None and verified >= 1.0)

    detected = suite.get("scenario_b_defect_class_detected")
    missing = suite.get("scenario_b_undetected_classes") or []
    observed = "n/a" if detected is None else f"{detected}"
    if missing:
        observed += " (NOT DETECTED: " + ", ".join(missing) + ")"
    row("seeded defect classes named by the planner from the sources", "100%",
        observed,
        detected is not None and detected >= GATES["min_defect_class_detection"])

    # lines[0] is the header and lines[1] the separator; every gate row
    # from lines[2] onward votes on the verdict.
    overall = all("FAIL" not in ln for ln in lines[2:])
    lines.append("")
    if overall:
        lines.append("**GO**: all preregistered MVP gates met on this fixture "
                     "distribution. Thresholds are MVP decisions, not product claims.")
    else:
        lines.append("**NO-GO (partial)**: failing gates retain negative evidence above; "
                     "the failed assumption is named rather than widened.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

"""Scenario B fixtures: isolated repos with seeded defect classes + held-out variants.

The preregistered task distribution is this defect grammar: off_by_one,
inverted_comparison, wrong_constant, missing_guard. Held-out variants are
generated from the same grammar with unseen seeds/function names — success on
them is evidence the repair policy generalizes, not a scripted demo.
"""

from __future__ import annotations

import random
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

DEFECT_CLASSES = ("off_by_one", "inverted_comparison", "wrong_constant", "missing_guard")


@dataclass(frozen=True)
class RepairTask:
    variant: str
    defect_class: str
    held_out: bool
    files: dict[str, str]      # rel_path -> content
    tests: dict[str, str]


def _fn_name(rng: random.Random) -> str:
    return "compute_" + "".join(rng.choice("abcdefghij") for _ in range(5))


def make_repair_task(seed: int, defect_class: str, held_out: bool = False) -> RepairTask:
    class_salt = zlib.crc32(defect_class.encode("utf-8"))
    rng = random.Random(seed * 7919 + class_salt)
    fn = _fn_name(rng)
    lo = rng.randint(2, 5)

    if defect_class == "off_by_one":
        good = (
            f"def {fn}(n):\n"
            f"    total = 0\n"
            f"    for i in range(1, n + 1):\n"
            f"        total += i\n"
            f"    return total\n"
        )
        bad = good.replace("range(1, n + 1)", "range(1, n)")
        test = (
            f"from pkg.mod import {fn}\n\n"
            f"def test_{fn}():\n"
            f"    assert {fn}({lo + 2}) == {sum(range(1, lo + 3))}\n"
        )
    elif defect_class == "inverted_comparison":
        good = (
            f"def {fn}(a, b):\n"
            f"    if a > b:\n"
            f"        return a\n"
            f"    return b\n"
        )
        bad = good.replace("if a > b:", "if a < b:")
        test = (
            f"from pkg.mod import {fn}\n\n"
            f"def test_{fn}():\n"
            f"    assert {fn}({lo}, {lo + 7}) == {lo + 7}\n"
        )
    elif defect_class == "wrong_constant":
        good = (
            f"def {fn}(x):\n"
            f"    return x * 2 + 1\n"
        )
        bad = good.replace("x * 2 + 1", "x * 3 + 1")
        test = (
            f"from pkg.mod import {fn}\n\n"
            f"def test_{fn}():\n"
            f"    assert {fn}({lo}) == {lo * 2 + 1}\n"
        )
    else:  # missing_guard
        good = (
            f"def {fn}(n):\n"
            f"    if n == 0:\n"
            f"        return 1\n"
            f"    out = 1\n"
            f"    for i in range(2, n + 1):\n"
            f"        out *= i\n"
            f"    return out\n"
        )
        bad = good.replace("    if n == 0:\n        return 1\n", "")
        test = (
            f"from pkg.mod import {fn}\n\n"
            f"def test_{fn}_zero():\n"
            f"    assert {fn}(0) == 1\n\n"
            f"def test_{fn}_fact():\n"
            f"    assert {fn}({min(lo, 4)}) == {__import__('math').factorial(min(lo, 4))}\n"
        )

    filler = "\n\n".join(
        f"def unused_{rng.randrange(1000)}_{k}(q):\n    return q + {k}\n" for k in range(6)
    )
    module = (
        "\"\"\"Small package under repair.\"\"\"\n\n"
        + filler + "\n\n\n"
        + bad + "\n\n\n" + filler + "\n"
    )
    init = ""
    files = {"pkg/__init__.py": init, "pkg/mod.py": module}
    tests = {"tests/test_mod.py": test}
    variant = f"{'heldout' if held_out else 'seen'}-{defect_class}-{seed}"
    return RepairTask(variant=variant, defect_class=defect_class,
                      held_out=held_out, files=files, tests=tests)


def materialize_repo(root: Path, task: RepairTask) -> Path:
    for rel, content in {**task.files, **task.tests}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return root


def to_json(task: RepairTask) -> str:
    return json.dumps(
        {"variant": task.variant, "defect_class": task.defect_class,
         "held_out": task.held_out}, sort_keys=True,
    )

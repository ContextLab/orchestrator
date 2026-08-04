#!/usr/bin/env python3
"""What the example catalogue currently does, grouped by why.

The catalogue is a mixture: pipelines that are part of the supported product,
pipelines written against superseded schemas, `enhanced/` twins of other
files, and pipelines referencing tools that do not exist. A single number --
"50 of 117 validate" -- hides all of that, and moves for reasons that are not
regressions: fixing a twin moves it by two, retiring a legacy file moves it by
one.

So this reports two different things, and only one of them can fail the build:

* **The validation baseline.** `scripts/catalogue_validation_baseline.txt`
  names the files that *pass `orchestrator validate`* today. A file dropping
  out of it is a regression and fails. A file newly validating is an
  improvement and does not -- it just prints a reminder to add it.

  It says nothing more than that. Passing validation is not the same as
  running, producing correct artifacts, or being a supported example, and
  calling this a "supported list" conflated four different things.
  `examples/supported/` is the stronger contract: those are executed and
  their behaviour asserted. This file is the weaker, wider net.

* **Everything else, grouped by the validator's own first error.** Counts by
  signature rather than by a taxonomy written down somewhere, because a
  hand-maintained taxonomy drifts from what the validator actually says and
  then quietly mis-describes the backlog.

Usage:
    python scripts/catalogue_report.py             # report; fail on regressions
    python scripts/catalogue_report.py --update    # rewrite the baseline
    python scripts/catalogue_report.py --json      # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
BASELINE = REPO / "scripts" / "catalogue_validation_baseline.txt"

#: Collapses the variable parts of an error so that "Tool 'a' not found" and
#: "Tool 'b' not found" are one signature rather than two.
_QUOTED = re.compile(r"'[^']*'")
_NUMBER = re.compile(r"\b\d+\b")
_BULLET = re.compile(r"^\s+-\s+(.*\S)\s*$")

#: Where the validator's summary list starts. Bullets before it are echoed
#: source, not findings.
_FAILED_HEADING = re.compile(r"validation failed", re.I)

#: `2026-08-03 19:52:19 - orchestrator.validation... - ERROR - <message>`.
#: The timestamp must never reach a signature: it would make every run report
#: a different set of groups.
_LOG = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^-]*- [\w.]+ - (?:ERROR|CRITICAL) - (.*)$"
)

#: The validator prefixes its messages with the phase that produced them.
_PHASE = re.compile(r"^Validation error in \w+:\s*")

#: A file that does not parse never reaches the validator, so its output is a
#: parser traceback plus echoed source lines -- which contain whatever words
#: the pipeline happened to use, including "error".
_UNPARSEABLE = re.compile(
    r"yaml\.(scanner|parser|composer|constructor)|"
    r"ScannerError|ParserError|ComposerError|"
    r"could not find expected|found character|mapping values are not allowed",
    re.I,
)


def example_files() -> List[Path]:
    return sorted(EXAMPLES.rglob("*.yaml"))


def validate(path: Path) -> Tuple[bool, str]:
    """Run `orchestrator validate` exactly as a user would."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    result = subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", "validate", str(path)],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300,
    )
    return result.returncode == 0, result.stdout + result.stderr


def first_error(output: str) -> str:
    """The validator's own first complaint, with the specifics removed.

    Prefers the bulleted list under "validation failed", because that is the
    validator speaking. Falls back to the first line mentioning a failure,
    which is what YAML parse errors and crashes produce -- they never reach
    the bulleted form, and 22 of the current failures are of that kind.
    """
    lines = output.splitlines()

    # 1. The validator's own summary list, when it got far enough to produce
    #    one. Only bullets *after* the "validation failed" heading count: a
    #    YAML list item in echoed source is also `  - something`, and matching
    #    those grouped one file under "id: legacy_tool_usage".
    start = next(
        (i for i, line in enumerate(lines) if _FAILED_HEADING.search(line)), None
    )
    if start is not None:
        for line in lines[start + 1:]:
            match = _BULLET.match(line)
            if match:
                return _signature(match.group(1))

    # 2. A file that does not parse never reaches the validator. Checked before
    #    the log scan because the parser echoes source lines, and a pipeline
    #    whose prose happens to contain "error" would otherwise be grouped by
    #    its own text.
    if _UNPARSEABLE.search(output):
        return "YAML does not parse"

    # 3. Otherwise the first logged error, without its timestamp.
    for line in lines:
        match = _LOG.match(line.strip())
        if match:
            return _signature(_PHASE.sub("", match.group(1)))

    return "(no error reported)"


def _signature(message: str) -> str:
    """One error, with its specifics removed so like groups with like.

    Numbers go too: "Schema validation failed: 56 errors" and "...: 4272
    errors" are one kind of problem, and leaving the count in split a single
    class across nine rows.
    """
    message = _QUOTED.sub("'...'", message.strip())
    message = _NUMBER.sub("N", message)
    return message[:90]


def compare(
    supported: List[str], validating: List[str]
) -> Tuple[List[str], List[str]]:
    """(files that stopped validating, files that newly validate).

    Only the first is a failure. The second is progress, and making it fail
    would mean every repaired example broke the build until someone updated a
    list -- which teaches people to stop repairing examples.
    """
    return (
        sorted(set(supported) - set(validating)),
        sorted(set(validating) - set(supported)),
    )


def load_baseline() -> Optional[List[str]]:
    if not BASELINE.exists():
        return None
    return [
        line.strip()
        for line in BASELINE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def write_baseline(paths: List[str]) -> None:
    BASELINE.write_text(
        "# Examples that pass `orchestrator validate`.\n"
        "#\n"
        "# A file leaving this list is a regression and fails CI. A file\n"
        "# joining it is an improvement; run:\n"
        "#     python scripts/catalogue_report.py --update\n"
        "#\n"
        "# The count is deliberately not the contract -- see the module\n"
        "# docstring in scripts/catalogue_report.py.\n"
        + "".join(f"{p}\n" for p in sorted(paths))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="rewrite the baseline from what validates now")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    files = example_files()
    if not files:
        print("no examples found", file=sys.stderr)
        return 1

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(validate, files))

    validating: List[str] = []
    failures: Dict[str, str] = {}
    for path, (ok, output) in zip(files, results):
        rel = str(path.relative_to(REPO))
        if ok:
            validating.append(rel)
        else:
            failures[rel] = first_error(output)

    if args.update:
        write_baseline(validating)
        print(f"wrote {BASELINE.relative_to(REPO)} with {len(validating)} entries")
        return 0

    signatures = Counter(failures.values())
    baseline = load_baseline()
    regressed, gained = compare(baseline or [], validating)

    if args.json:
        # Nothing but JSON on stdout: this used to print the human report
        # afterwards, so anything parsing it hit "Extra data".
        print(json.dumps({
            "total": len(files),
            "validating": len(validating),
            "failing": len(failures),
            "by_signature": dict(signatures.most_common()),
            "failures": failures,
            "regressed": regressed,
            "gained": gained if baseline is not None else [],
        }, indent=2))
        return 1 if (baseline is not None and regressed) else 0

    print(f"catalogue: {len(validating)}/{len(files)} validate\n")
    print(f"{'count':>5}  first error")
    print(f"{'-' * 5}  {'-' * 60}")
    for signature, count in signatures.most_common():
        print(f"{count:>5}  {signature}")

    if baseline is None:
        print(f"\nno {BASELINE.relative_to(REPO)} yet; create it with --update")
        return 0

    if gained:
        print(f"\n{len(gained)} example(s) now validate and are not yet listed:")
        for path in gained:
            print(f"  + {path}")
        print("  run: python scripts/catalogue_report.py --update")

    if regressed:
        print(f"\n{len(regressed)} example(s) stopped validating:")
        for path in regressed:
            print(f"  - {path}: {failures.get(path, 'unknown')}")
        print("\nThese are listed as working, so this is a regression. The "
              "headline count is not the contract; these named files are.")
        return 1

    print(f"\nall {len(baseline)} listed example(s) still validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())

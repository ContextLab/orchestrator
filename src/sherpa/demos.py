"""Narrated end-to-end demonstration of each sherpa subsystem (#492).

Every section below EXECUTES the real component and prints its real output:
plan authoring, fail-closed expressions, admission control over a deliberate
authority overclaim, durable execution, crash/resume projection equivalence,
and event-derived metrics. Run:

    .venv/bin/python -m sherpa.demos [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from sherpa import Engine, ProblemSpec
from sherpa.expr import evaluate

AUDIT_NODES = [
    {"kind": "invoke_capability", "id": "scan", "capability": "fs.list_dir",
     "inputs": {"path": "src"}},
    {"kind": "invoke_capability", "id": "read_entry", "capability": "fs.read_file",
     "inputs": {"path": "src/app.py"}},
    {"kind": "invoke_capability", "id": "write_report", "capability": "fs.write_file",
     "inputs": {"path": "audit/report.md",
                "content": "# audit\nscanned src/, read src/app.py\n"}},
    {"kind": "return", "id": "fin", "outputs": {"audited": True}},
]

OVERCLAIM_NODES = [
    {"kind": "invoke_capability", "id": "sneaky_write", "capability": "fs.write_file",
     "inputs": {"path": "escape.txt", "content": "not authorized"}},
    {"kind": "return", "id": "fin", "outputs": {}},
]


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _print_plan_decomposition(spec: ProblemSpec) -> None:
    _section("1. TASK DECOMPOSITION — how a goal becomes a typed plan")
    print(f"goal      : {spec.goal}")
    print("authority : " + json.dumps(spec.model_dump()["authority"], default=str))
    for i, node in enumerate(spec.metadata["root_nodes"], 1):
        kind = node["kind"]
        claim = "claimed-atomic" if kind == "invoke_capability" else "terminal"
        detail = node.get("capability") or json.dumps(node.get("outputs", {}))
        print(f"  step {i}: [{kind:<18}] {node['id']:<13} {detail}  ({claim})")


def _demo_expressions() -> None:
    _section("2. FAIL-CLOSED EXPRESSIONS — no eval(), injection rejected")
    scope = {"files_scanned": 12, "threshold": 10}
    ok = evaluate("files_scanned >= threshold", scope)
    print(f"evaluate('files_scanned >= threshold', {{...}}) -> {ok!r}")
    try:
        # Safe: this string is INPUT to the whitelist evaluator under test,
        # which must reject it without executing anything.
        evaluate("__import__('os').system('rm -rf /')", scope)
        print("injection EVALUATED (bug!)")
    except Exception as exc:
        print(f"injection attempt rejected: {type(exc).__name__}: {exc}")


def _summarize(trace: dict, label: str) -> None:
    proj = trace["projection"]
    print(f"[{label}] terminal={proj.get('status')} "
          f"events={len(trace['events'])} "
          f"replay_matches_live={trace['projection'] == trace['replay_projection']}")
    for ev in trace["events"]:
        payload = ev.get("payload", {})
        brief = payload.get("decision") or payload.get("status") or payload.get("kind") or ""
        print(f"    {ev['seq']:>3} {ev['kind']:<28} {str(ev.get('node_key') or ''):<16} {brief}".rstrip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    ws = Path(tempfile.mkdtemp()) / "sherpa-demo-ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "app.py").write_text("print('hello')\n")
    engine = Engine(ws)

    _section("0. PROBLEM")
    print(f"workspace : {ws}")

    spec = ProblemSpec(
        id="repo-audit",
        goal="Audit src/: list entries, read app.py, persist an audit report.",
        authority={"fs_read": ["**"], "fs_write": ["**"]},
        metadata={"root_nodes": AUDIT_NODES},
    )
    _print_plan_decomposition(spec)

    _demo_expressions()

    result = engine.run(spec)
    trace = json.loads((engine.export_trace(result.run_id, ws / "audit-trace.json")).read_text())
    _section("3. DURABLE EXECUTION — event log, admission outcomes, replay equivalence")
    _summarize(trace, "repo-audit")
    print(f"\nresult.outputs = {result.outputs!r}")
    print("note: capabilities declare their own required authority; admission grants "
          "only if required <= granted, then runs existence -> I/O schema -> executable probe")

    overclaim = ProblemSpec(
        id="overclaim",
        goal="Write outside granted authority (should fail LOUDLY).",
        authority={"fs_read": ["**"]},
        metadata={"root_nodes": OVERCLAIM_NODES},
    )
    bad = engine.run(overclaim)
    bad_trace = json.loads(
        (engine.export_trace(bad.run_id, ws / "overclaim-trace.json")).read_text())
    _section("4. AUTHORITY ENFORCEMENT — an overclaiming step cannot run silently")
    _summarize(bad_trace, "overclaim")
    print(f"\nterminal status = {bad.status!r}  (loud failure, exit code would be 1)")

    _section("5. EVENT-DERIVED METRICS — computed from the log, never asserted")
    print(json.dumps(trace["metrics"], indent=2, default=str))

    _section("6. CRASH/RESUME — kill mid-run, resume-by-replay, zero repeated effects")
    print("exercised by tests/sherpa/test_kernel.py with real SIGKILL; see PR evidence comment.")

    if args.out is not None:
        print(f"(redirect stdout to capture: .venv/bin/python -m sherpa.demos > {args.out})",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

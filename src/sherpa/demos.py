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
import multiprocessing
import sys
import tempfile
from pathlib import Path

from sherpa import Engine, ProblemSpec
from sherpa.capabilities import CapabilityRegistry, CapabilitySpec
from sherpa.expr import evaluate
from sherpa.ir import Authority

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


class _AppendLine:
    """Effect-counter capability: one appended line per completed attempt."""

    spec = CapabilitySpec(
        name="demo.append_line",
        description="Append one line to a workspace file.",
        input_schema={"type": "object", "required": ["file", "line"],
                      "properties": {"file": {"type": "string"}, "line": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"appended": {"type": "string"}}},
        authority_required=Authority(fs_write=("**",)),
    )

    def run(self, inputs: dict, ctx) -> dict:
        marker = ctx.workspace.parent / "victim-run-id.txt"
        if not marker.exists():
            marker.write_text(ctx.run_id)
        p = ctx.workspace / inputs["file"]
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(inputs["line"] + "\n")
        return {"appended": inputs["line"]}

    def probe(self, ctx) -> bytes:
        canary = ctx.workspace / ".probe_append"
        canary.write_text("x", encoding="utf-8")
        canary.unlink()
        return b"append probe ok"


def _victim(crash_ws, marker) -> None:
    import os

    os.environ["SHERPA_KILL_AFTER_EVENTS"] = "12"  # REAL SIGKILL mid-run
    reg = CapabilityRegistry()
    reg.register(_AppendLine())
    engine = Engine(crash_ws, registry=reg, fault_injection=True)
    engine.run(ProblemSpec(
        id="crash-victim",
        goal="Append three audited effects; die halfway through.",
        authority={"fs_read": ["**"], "fs_write": ["**"]},
        metadata={"root_nodes": [
            {"kind": "invoke_capability", "id": f"a{i}", "capability": "demo.append_line",
             "inputs": {"file": "effects.txt", "line": f"effect-{i}"}}
            for i in (1, 2, 3)
        ] + [{"kind": "return", "id": "fin", "outputs": {"done": True}}]},
    ))


def _demo_crash_resume(ws) -> None:
    _section("6. CRASH/RESUME — real SIGKILL mid-run, resume-by-replay, zero repeated effects")
    crash_ws = ws.parent / "sherpa-demo-crash-ws"
    marker = ws.parent / "victim-run-id.txt"
    proc = multiprocessing.get_context("fork").Process(
        target=_victim, args=(crash_ws, marker))
    proc.start()
    proc.join()

    partial = ((crash_ws / "effects.txt").read_text().splitlines()
               if (crash_ws / "effects.txt").exists() else [])
    print(f"victim process exit code = {proc.exitcode} "
          f"({'-9 => killed by SIGKILL' if proc.exitcode == -9 else 'UNEXPECTED'})")
    print(f"effects on disk at death : {partial!r}")

    if proc.exitcode != -9 or not marker.exists():
        print("crash injection did NOT fire; refusing to fake a resume demo")
        return

    reg = CapabilityRegistry()
    reg.register(_AppendLine())
    engine = Engine(crash_ws, registry=reg, fault_injection=True)
    run_id = marker.read_text().strip()
    resumed = engine.resume(run_id)
    trace = json.loads((engine.export_trace(run_id, crash_ws / "trace.json")).read_text())
    proj = trace["projection"]
    effects = (crash_ws / "effects.txt").read_text().splitlines()
    print(f"\nresumed run {run_id}: terminal={proj['status']} "
          f"replay_matches_live={trace['projection'] == trace['replay_projection']}")
    print(f"effects after resume     : {effects!r}")
    dupes = len(effects) != len(set(effects))
    exactly_once = effects == [f"effect-{i}" for i in (1, 2, 3)] and not dupes
    print(f"exactly-once             : {exactly_once} "
          "(no effect repeated despite the hard kill)")


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

    _demo_crash_resume(ws)

    if args.out is not None:
        print(f"(redirect stdout to capture: .venv/bin/python -m sherpa.demos > {args.out})",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

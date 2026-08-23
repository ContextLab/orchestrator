"""Kernel tests: durable execution, SIGKILL crash/resume, budgets, gates.

The crash test kills a REAL child process with SIGKILL mid-run and resumes it
in a fresh Engine — the issue's durability fixture in miniature. No mocks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sherpa.capabilities import Capability, CapabilityContext, CapabilityRegistry, CapabilitySpec
from sherpa.ir import Authority, Budgets, ProblemSpec
from sherpa.kernel import Engine, RunResult

pytestmark = [pytest.mark.unit]


class EchoWrite(Capability):
    spec = CapabilitySpec(
        name="demo.append_line",
        input_schema={"type": "object", "required": ["file", "line"],
                      "properties": {"file": {"type": "string"}, "line": {"type": "string"}}},
        output_schema={"type": "object"},
        authority_required=Authority(fs_write=("**",)),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        p = ctx.workspace / inputs["file"]
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(inputs["line"] + "\n")
        return {"appended": inputs["line"], "file": inputs["file"]}

    def probe(self, ctx: CapabilityContext) -> bytes:
        canary = ctx.workspace / ".probe_append"
        canary.write_text("x", encoding="utf-8")
        canary.unlink()
        return b"append probe ok"


def _registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(EchoWrite())
    return reg


def _engine(workspace: Path) -> Engine:
    return Engine(workspace, registry=_registry())


FULL_AUTH = Authority(fs_read=("**",), fs_write=("**",), subprocess_allow=("**",))


def _linear_problem(tmp: Path) -> ProblemSpec:
    return ProblemSpec(
        id="linear-demo",
        goal="append three audited lines",
        authority=FULL_AUTH,
        metadata={
            "root_nodes": [
                {"kind": "invoke_capability", "id": "w1", "capability": "demo.append_line",
                 "inputs": {"file": "out.txt", "line": "one"}},
                {"kind": "invoke_capability", "id": "w2", "capability": "demo.append_line",
                 "inputs": {"file": "out.txt", "line": "two"}},
                {"kind": "return", "id": "fin",
                 "outputs": {"lines_file": "{{ w1.result.file }}"}},
            ],
        },
    )


class TestHappyPath:
    def test_linear_plan_completes(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        result = eng.run(_linear_problem(tmp_path))
        assert result.status == "completed"
        assert (tmp_path / "ws" / "out.txt").read_text() == "one\ntwo\n"
        kinds = [e.kind for e in eng.store.events(run_id=result.run_id)]
        assert "admission_checked" in kinds and "tool_call_finished" in kinds

    def test_branch_selects_on_predicate(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")

        def spec(mode: str) -> ProblemSpec:
            return ProblemSpec(
                id="branch-demo",
                goal="branch by mode",
                inputs={"mode": mode},
                authority=FULL_AUTH,
                metadata={
                    "root_nodes": [
                        {"kind": "branch", "id": "route", "cases": [
                            {"when": "inputs.mode == 'fast'",
                             "body": [{"kind": "invoke_capability", "id": "fast",
                                       "capability": "demo.append_line",
                                       "inputs": {"file": "path.txt", "line": "fast"}}]},
                            {"when": None,
                             "body": [{"kind": "invoke_capability", "id": "slow",
                                       "capability": "demo.append_line",
                                       "inputs": {"file": "path.txt", "line": "slow"}}]},
                        ]},
                        {"kind": "return", "id": "fin", "outputs": {}},
                    ],
                },
            )

        fast = eng.run(spec("fast"))
        assert fast.status == "completed"
        assert (tmp_path / "ws" / "path.txt").read_text() == "fast\n"

        ws2 = tmp_path / "ws2"
        slow_engine = _engine(ws2)
        slow = slow_engine.run(spec("careful"))
        assert slow.status == "completed"
        assert (ws2 / "path.txt").read_text() == "slow\n"


def _while_problem() -> ProblemSpec:
    """While-loop driven by an event-sourced counter capability-free predicate.

    The guard reads scope['inputs']['limit'] vs loop iterations via journal —
    but guards only see scope; so we model counting through repeated appends
    bounded by max_iterations and a guard that stays true until file grows.
    Simplest deterministic version: guard 'inputs.keep_going' with limit 3.
    """
    return ProblemSpec(
        id="while-demo",
        goal="bounded ticking",
        inputs={"keep_going": True},
        authority=FULL_AUTH,
        metadata={
            "root_nodes": [
                {"kind": "while", "id": "loop", "guard": "inputs.keep_going",
                 "max_iterations": 3,
                 "body": [{"kind": "invoke_capability", "id": "tick",
                           "capability": "demo.append_line",
                           "inputs": {"file": "ticks.txt", "line": "ticked"}}]},
                {"kind": "return", "id": "fin", "outputs": {"done": True}},
            ],
        },
    )


class TestBoundedLoop:
    def test_while_respects_max_iterations(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        result = eng.run(_while_problem())
        assert result.status == "completed"
        text = (tmp_path / "ws" / "ticks.txt").read_text()
        assert text.count("ticked") == 3


class TestCrashResume:
    def test_sigkill_midrun_then_resume_equivalent(self, tmp_path: Path) -> None:
        """REAL SIGKILL during execution; resume completes without repeating work."""
        ws = tmp_path / "ws"
        problem = _linear_problem(tmp_path)
        repo_root = Path(__file__).parent.parent.parent
        code = (
            "import json,sys;"
            f"sys.path.insert(0,{json.dumps(str(repo_root / 'src'))});"
            f"sys.path.insert(0,{json.dumps(str(repo_root / 'tests' / 'sherpa'))});"
            "from pathlib import Path;"
            "from kernel_subprocess_support import build_spec, build_engine, kill_self_after;"
            f"spec=build_spec({json.dumps(problem.model_dump_json())});"
            f"eng=build_engine(Path({json.dumps(str(ws))}));"
            "kill_self_after(14);"
            "r=eng.run(spec);"
            "print(json.dumps({'status': r.status, 'run_id': r.run_id}))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ},
            timeout=120,
        )
        assert proc.returncode == -9, (
            f"child should die by SIGKILL, got rc={proc.returncode}; "
            f"stderr tail: {proc.stderr[-500:]}"
        )

        eng2 = _engine(ws)
        rows = eng2.store.conn.execute("SELECT run_id FROM runs").fetchall()
        assert rows, "interrupted run must be durably recorded"
        rid = rows[-1]["run_id"]
        before_events = len(eng2.store.events(run_id=rid))

        result2 = eng2.resume(rid)
        assert result2.status == "completed"
        out_txt = (ws / "out.txt").read_text()
        assert out_txt == "one\ntwo\n", f"exactly-once effects violated: {out_txt!r}"
        after_events = len(eng2.store.events(run_id=rid))
        assert after_events > before_events

        ref_ws = tmp_path / "ref_ws"
        eng_ref = _engine(ref_ws)
        ref = eng_ref.run(problem)

        # Lineage/projection EQUIVALENCE (issue A): same loud outcome, same node
        # end-states, same outputs. Event interleaving may differ around the crash.
        live_a = eng2.store.projection(rid)
        assert eng2.store.replay_projection(rid)["status"] == "completed"
        ref_proj = eng_ref.store.projection(ref.run_id)
        assert live_a["status"] == ref_proj["status"] == "completed"
        ref_node_states = {k: v["state"] for k, v in ref_proj["nodes"].items()}
        res_node_states = {
            k.rsplit("@", 1)[0]: v["state"]
            for k, v in live_a["nodes"].items()
            if v["state"] != "skipped"
        }
        assert res_node_states == ref_node_states
        assert result2.outputs == ref.outputs

    def test_resume_of_completed_run_is_noop(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        result = eng.run(_linear_problem(tmp_path))
        again = eng.resume(result.run_id)
        assert again.status == "completed"


class TestBudgetsAndTerminals:
    def test_budget_exhausted_is_loud(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        spec = _linear_problem(tmp_path)
        spec.budgets.max_nodes = 1  # force exhaustion before second capability node
        result = eng.run(spec)
        assert result.status == "budget_exhausted"

    def test_intentional_fail_node_loud(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        spec = ProblemSpec(
            id="fail-demo",
            goal="fail on purpose",
            metadata={"root_nodes": [
                {"kind": "fail", "id": "boom", "reason": "seeded failure"},
            ]},
        )
        result = eng.run(spec)
        assert result.status == "failed" and "seeded failure" in (result.error or "")

    def test_ask_user_child_escalates(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        spec = ProblemSpec(
            id="ask-demo",
            goal="need input",
            attended=False,
            metadata={"root_nodes": [
                {"kind": "ask_user", "id": "ask", "question": "which way?"},
                {"kind": "return", "id": "fin", "outputs": {}},
            ]},
        )
        result = eng.run(spec)
        assert result.status == "escalated"

    def test_attended_root_blocks_then_message_resumes(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        spec = ProblemSpec(
            id="attend-demo",
            goal="need input",
            attended=True,
            metadata={"root_nodes": [
                {"kind": "ask_user", "id": "ask", "question": "continue?"},
                {"kind": "return", "id": "fin", "outputs": {"answer_given": True}},
            ]},
        )
        blocked = eng.run(spec)
        assert blocked.status == "blocked"
        eng.deliver_message(blocked.run_id, next(iter(
            e.node_key for e in eng.store.events(run_id=blocked.run_id)
            if e.payload.get("new") == "blocked" and e.node_key
        )), {"choice": "yes"})
        resumed = eng.resume(blocked.run_id)
        assert resumed.status == "completed"
        assert resumed.outputs.get("answer_given") is True

    def test_admission_escalates_unknown_capability(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        spec = ProblemSpec(
            id="ghost-demo",
            goal="call unregistered capability",
            metadata={"root_nodes": [
                {"kind": "invoke_capability", "id": "g", "capability": "ghost.op"},
                {"kind": "return", "id": "fin", "outputs": {}},
            ]},
        )
        result = eng.run(spec)
        assert result.status == "escalated"
        assert "not registered" in (result.error or "")

    def test_final_review_blocks_bad_output(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        spec = ProblemSpec(
            id="gate-demo",
            goal="predicate must hold",
            acceptance=[{
                "id": "must_be_positive",
                "kind": "predicate",
                "spec": {"expr": "outputs.value > 0"},
            }],
            metadata={"root_nodes": [
                {"kind": "return", "id": "fin", "outputs": {"value": -5}},
            ]},
        )
        result = eng.run(spec)
        assert result.status == "escalated"
        assert "final review blocked" in (result.error or "")

    def test_predicate_acceptance_passes_good_output(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path / "ws")
        spec = ProblemSpec(
            id="gate-ok",
            goal="predicate holds",
            acceptance=[{
                "id": "must_be_positive",
                "kind": "predicate",
                "spec": {"expr": "outputs.value > 0"},
            }],
            metadata={"root_nodes": [
                {"kind": "return", "id": "fin", "outputs": {"value": 7}},
            ]},
        )
        result = eng.run(spec)
        assert result.status == "completed"

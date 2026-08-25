"""Kernel guarantees that the audit found unmet: exactly-once execution,
runtime recursion depth, persisted parent/child linkage, and loud terminals.

Each test corresponds to a defect demonstrated against the pre-fix tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherpa.capabilities import Capability, CapabilitySpec, ProbeFailed
from sherpa.ir import Budgets, Plan, ProblemSpec
from sherpa.kernel import Engine

# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


def _child_plan(plan_id: str, marker: str, nested_subgoal: str | None = None,
                library: list | None = None) -> dict:
    """A child plan that writes a file, optionally decomposing once more.

    ``library`` is threaded into the nested decompose's hints because
    ``StubPlanner`` resolves subgoals from ``hints["plan_library"]`` and the
    kernel does not propagate a library down the tree -- each decompose must
    carry the fixture it needs.
    """
    body: list[dict] = [
        {
            "kind": "invoke_capability",
            "id": "w",
            "capability": "fs.write_file",
            "inputs": {"path": f"{marker}.txt", "content": marker},
        }
    ]
    if nested_subgoal:
        body.append({
            "kind": "decompose", "id": "deeper", "subgoal": nested_subgoal,
            "hints": {"plan_library": library if library is not None else []},
        })
    body.append({"kind": "return", "id": "fin", "outputs": {"marker": marker}})
    return {"id": plan_id, "authority": {}, "budgets": {"max_fanout": 2}, "root": body}


def _library(entries: list[tuple[str, dict]]) -> list[dict]:
    return [{"match": {"pattern_contains": pat}, "plan": plan} for pat, plan in entries]


def _spec(spec_id: str, root_nodes: list[dict], **kw) -> ProblemSpec:
    return ProblemSpec(
        id=spec_id,
        goal=kw.pop("goal", "kernel durability probe"),
        authority={"fs_read": ["**"], "fs_write": ["**"]},
        metadata={"root_nodes": root_nodes},
        **kw,
    )


# --------------------------------------------------------------------------
# exactly-once: a lease we could not acquire must stop us
# --------------------------------------------------------------------------


def test_node_is_not_executed_when_its_lease_is_held_elsewhere(tmp_path: Path) -> None:
    """``_begin_attempt`` returned True when ``acquire_lease`` FAILED, so a
    second worker executed a node another worker already held. The lease itself
    was correct; the kernel ignored its answer."""
    ws = tmp_path / "ws"
    engine = Engine(ws)
    rid = "run_leased"
    engine.store.create_run(rid, "sha")
    node_key = "plan.node"
    engine.store.upsert_node(rid, node_key, "pending", depth=0)
    assert engine.store.acquire_lease(rid, node_key, "other_live_session") is True

    began = engine._begin_attempt(rid, node_key, "our_session", 0)
    assert began is False, "kernel proceeded to execute a node leased by another session"
    engine.close()


def test_effects_are_not_repeated_when_a_completed_run_is_resumed(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    target = "effect.txt"
    engine = Engine(ws)
    spec = _spec(
        "resume_noop",
        [
            {
                "kind": "invoke_capability",
                "id": "w",
                "capability": "fs.write_file",
                "inputs": {"path": target, "content": "once"},
            },
            {"kind": "return", "id": "fin", "outputs": {"ok": True}},
        ],
    )
    first = engine.run(spec)
    assert first.status == "completed"
    head_before = engine.store.head_seq()

    again = engine.resume(first.run_id)
    assert again.status == "completed"
    assert engine.store.head_seq() == head_before, "resume of a completed run did work"
    engine.close()


# --------------------------------------------------------------------------
# recursion depth is real state, not a permanently-zero local
# --------------------------------------------------------------------------


def test_decomposition_depth_is_recorded_per_node(tmp_path: Path) -> None:
    """``depth`` was initialised to 0 in ``_drive`` and never incremented, and
    ``_author_child``'s ``depth`` parameter was dead, so every node in the
    ``nodes`` table recorded depth 0 no matter how deep it actually was."""
    ws = tmp_path / "ws"
    engine = Engine(ws)
    # Built bottom-up so each level carries only the level below it; a
    # self-referencing library would be a cycle and fail to serialize.
    lib_l2 = _library([("level two", _child_plan("plan_l2", "l2"))])
    library = _library([
        ("level one", _child_plan("plan_l1", "l1", nested_subgoal="level two",
                                  library=lib_l2)),
    ])
    spec = _spec(
        "depths",
        [
            {"kind": "decompose", "id": "d1", "subgoal": "level one",
             "hints": {"plan_library": library}},
            {"kind": "return", "id": "fin", "outputs": {"ok": True}},
        ],
    )
    result = engine.run(spec)
    assert result.status == "completed", result.error
    depths = {
        key: node["depth"] for key, node in engine.store.projection(result.run_id)["nodes"].items()
    }
    assert max(depths.values()) >= 2, f"nested decomposition recorded no depth: {depths}"
    engine.close()


def test_max_depth_is_enforced_at_runtime(tmp_path: Path) -> None:
    """``budgets.max_depth`` only constrained the *structural* nesting of
    branch/while/parallel inside one plan; recursive decomposition was bounded
    only by wall-clock and node count."""
    ws = tmp_path / "ws"
    engine = Engine(ws)
    lib_l3 = _library([("level three", _child_plan("plan_l3", "l3"))])
    lib_l2 = _library([
        ("level two", _child_plan("plan_l2", "l2", nested_subgoal="level three",
                                  library=lib_l3)),
    ])
    library = _library([
        ("level one", _child_plan("plan_l1", "l1", nested_subgoal="level two",
                                  library=lib_l2)),
    ])
    spec = _spec(
        "depthcap",
        [
            {"kind": "decompose", "id": "d1", "subgoal": "level one",
             "hints": {"plan_library": library}},
            {"kind": "return", "id": "fin", "outputs": {"ok": True}},
        ],
        budgets=Budgets(max_depth=1),
    )
    result = engine.run(spec)
    assert result.status != "completed", "exceeding max_depth completed silently"
    assert result.status in ("budget_exhausted", "escalated", "blocked", "failed")
    engine.close()


# --------------------------------------------------------------------------
# the organization tree must be persisted, not held in a transient dict
# --------------------------------------------------------------------------


def test_parent_child_linkage_is_persisted(tmp_path: Path) -> None:
    """``parent_key`` exists in the schema but the kernel never passed it, so
    every node had ``parent_key IS NULL`` even after a real decomposition. The
    parent/child relation lived only in an in-memory dict discarded at exit."""
    ws = tmp_path / "ws"
    engine = Engine(ws)
    library = _library([("do the sub-task", _child_plan("plan_child", "childout"))])
    spec = _spec(
        "orgtree",
        [
            {"kind": "decompose", "id": "d1", "subgoal": "do the sub-task",
             "hints": {"plan_library": library}},
            {"kind": "return", "id": "fin", "outputs": {"ok": True}},
        ],
    )
    result = engine.run(spec)
    assert result.status == "completed", result.error

    nodes = engine.store.projection(result.run_id)["nodes"]
    parents = {key: node.get("parent_key") for key, node in nodes.items()}
    linked = {k: v for k, v in parents.items() if v}
    assert linked, f"no node recorded a parent_key; org tree is not persisted: {parents}"

    child_key = next(k for k in nodes if k.startswith("plan_child."))
    assert parents[child_key], f"child node {child_key} has no parent link"
    assert parents[child_key] in nodes, "parent_key does not reference a real node"
    engine.close()


def test_every_executed_node_records_its_owner_session(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    engine = Engine(ws)
    spec = _spec(
        "sessions",
        [
            {"kind": "invoke_capability", "id": "w", "capability": "fs.write_file",
             "inputs": {"path": "s.txt", "content": "x"}},
            {"kind": "return", "id": "fin", "outputs": {"ok": True}},
        ],
    )
    result = engine.run(spec)
    assert result.status == "completed", result.error
    nodes = engine.store.projection(result.run_id)["nodes"]
    executed = [n for k, n in nodes.items() if k.endswith(".w")]
    assert executed, "capability node missing from the projection"
    assert all(n.get("owner_session") for n in executed), f"no owner_session recorded: {executed}"
    engine.close()


# --------------------------------------------------------------------------
# loud terminals: no run may be abandoned in `running`
# --------------------------------------------------------------------------


def test_planner_failure_is_a_loud_terminal_not_an_escaped_exception(tmp_path: Path) -> None:
    """A planner miss raised ``NoPlanTemplate`` straight out of ``Engine.run``:
    the run stayed ``running``, the node stayed ``pending``, and NO
    ``run_terminal`` event was ever written."""
    ws = tmp_path / "ws"
    engine = Engine(ws)
    spec = _spec(
        "plannermiss",
        [
            {"kind": "decompose", "id": "d1", "subgoal": "nothing in the library matches this",
             "hints": {"plan_library": _library([("something else", _child_plan("p", "m"))])}},
            {"kind": "return", "id": "fin", "outputs": {"ok": True}},
        ],
    )
    result = engine.run(spec)
    assert result.status in ("escalated", "blocked", "failed"), result.status
    proj = engine.store.projection(result.run_id)
    assert proj["status"] != "running", "run was abandoned in `running`"
    assert result.error, "loud terminal carried no explanation"
    engine.close()


def test_reclassified_atomic_claim_records_a_decompose_outcome(tmp_path: Path) -> None:
    """``decompose_outcome`` was emitted only when the parent was not
    ``skipped`` -- but the reclassify path sets it to ``skipped``, so every
    admission CORRECTION was structurally excluded from the metrics.

    The reclassify path builds its own hints and carries no plan library, so a
    real ``Planner`` implementation supplies the fallback plan here. This is a
    genuine implementation of the documented ``Planner`` protocol, not a mock:
    it authors and returns a real, validated ``Plan``.
    """

    class AlwaysProbeFails(Capability):
        spec = CapabilitySpec(
            name="test.unprobeable",
            description="A capability whose executable evidence cannot be produced.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

        def run(self, inputs: dict, ctx) -> dict:  # pragma: no cover - never admitted
            raise AssertionError("must never run: admission rejected the atomic claim")

        def probe(self, ctx) -> bytes:
            raise ProbeFailed("probe deliberately fails")

    class FallbackPlanner:
        """Authors one fixed fallback plan for any goal."""

        def __init__(self) -> None:
            self.goals: list[str] = []

        def author_plan(self, goal, hints, granted, budgets, session):
            self.goals.append(goal)
            return Plan(**_child_plan("plan_fallback", "fallback"))

    planner = FallbackPlanner()
    engine = Engine(tmp_path / "ws", planner=planner)
    engine.registry.register(AlwaysProbeFails())
    spec = _spec(
        "reclassify",
        [
            {"kind": "invoke_capability", "id": "u", "capability": "test.unprobeable",
             "inputs": {}, "atomic_claim": True},
            {"kind": "return", "id": "fin", "outputs": {"ok": True}},
        ],
    )
    result = engine.run(spec)

    events = engine.store.events(run_id=result.run_id)
    kinds = [e.kind for e in events]
    decisions = [e.payload.get("decision")
                 for e in events if e.kind == "admission_checked"]
    assert "reclassify_decompose" in decisions, f"probe failure did not reclassify: {decisions}"
    assert planner.goals, "reclassification never asked the planner for a fallback plan"
    assert "decompose_outcome" in kinds, (
        "a reclassified atomic claim emitted no decompose_outcome, so the corrected "
        f"branching factor cannot see it; events were {sorted(set(kinds))}"
    )
    outcomes = [e for e in events if e.kind == "decompose_outcome"]
    # The root plan emits one too (it is itself a decomposition of the goal),
    # so select the outcome for the node admission actually reclassified.
    reclassified = [e for e in outcomes if e.payload.get("reclassified")]
    assert reclassified, (
        f"no decompose_outcome recorded the admission correction; "
        f"outcomes were {[(e.node_key, e.payload.get('reclassified')) for e in outcomes]}"
    )
    assert reclassified[0].node_key.endswith(".u"), (
        f"outcome keyed to {reclassified[0].node_key!r}, not the reclassified node"
    )
    engine.close()


# --------------------------------------------------------------------------
# budgets must fail closed
# --------------------------------------------------------------------------


def test_negative_budgets_are_rejected(tmp_path: Path) -> None:
    for kwargs in (
        {"max_nodes": -5},
        {"max_depth": -1},
        {"max_fanout": -3},
        {"max_tokens": -1},
        {"max_wall_seconds": -9.0},
        {"max_attempts_per_node": 0},
    ):
        with pytest.raises(Exception):
            Budgets(**kwargs)


def test_cost_ceiling_is_not_fail_open(tmp_path: Path) -> None:
    """``max_cost_usd`` defaulted to 0.0 and was only consulted when
    ``> 0``, so the default meant *unlimited spend*, not *no spend*."""
    import inspect

    from sherpa import kernel as kernel_module

    source = inspect.getsource(kernel_module.Engine._check_budgets)
    assert "max_cost_usd > 0 and" not in source, (
        "cost ceiling is still skipped when the budget is 0, which makes the "
        "default unlimited rather than zero"
    )


# --------------------------------------------------------------------------
# model spend must be recorded, or token/cost budgets cannot bind
# --------------------------------------------------------------------------


def test_model_token_spend_is_recorded_against_the_run(tmp_path: Path) -> None:
    """`LiveChannel` always reported 0 tokens and `text.summarize` discarded
    the response's usage fields, so `Budgets.max_tokens` and `max_cost_usd`
    could never bind against real model spend."""
    from sherpa.channel import RecordedChannel

    engine = Engine(tmp_path / "ws")
    engine.channel = RecordedChannel({
        "summarizer": ["a summary of the corpus"],
        # admission's probe is a separate session, so it needs its own tape
        "summarizer_probe": ["pong"],
    })
    spec = _spec(
        "tokenspend",
        [
            {"kind": "invoke_capability", "id": "s", "capability": "text.summarize",
             "inputs": {"text": "some source material that needs summarizing"}},
            {"kind": "return", "id": "fin", "outputs": {"ok": True}},
        ],
    )
    result = engine.run(spec)
    assert result.status == "completed", result.error
    usage = engine.store.usage(result.run_id)
    assert usage["tokens"] > 0, f"model call recorded no token spend: {usage}"
    engine.close()


def test_estimated_tokens_are_flagged_not_passed_off_as_measured() -> None:
    from sherpa.channel import ChannelResponse, estimate_tokens

    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("x" * 400) == 100
    measured = ChannelResponse(text="hi", model="m", prompt_tokens=7, completion_tokens=3)
    assert measured.tokens_estimated is False
    assert measured.total_tokens == 10

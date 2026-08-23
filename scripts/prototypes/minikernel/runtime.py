"""The durable executor.

Everything the runtime knows is derived from the event log. There is no
in-memory tree of live agent objects: a logical node may be executed by many
sessions over its lifetime (retry, resume, model fallback, context rollover),
and one session may execute many nodes. `node == agent instance`, as #485
phrases it, is too rigid to survive a crash.

The four properties this module exists to demonstrate:

  P1  Durable nested runs. A child plan gets its own run_id with a recorded
      parent_run_id -- not an Orchestrator constructed inside a tool.
  P2  Resume. Kill the process mid-run; replay the log; completed work is
      skipped and the projection of the resumed run equals the projection of an
      uninterrupted one.
  P3  Loud exhaustion. A node that runs out of budget escalates to its parent
      with partial results and an explicit status. It never returns a green
      checkmark over truncated work.
  P4  Messages at deterministic boundaries. A message addressed to a pending
      node is delivered when that node starts, as a recorded event that mutates
      its inputs -- not as tokens injected into a live prompt.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .capabilities import CapabilityRegistry
from .ir import (Authority, Budget, Plan, PlanInvalid, Step, as_jsonable,
                 validate)
from .library import SolvedProblemLibrary
from .planner import PlanDraft, Planner
from .review import ArtifactVersion, Criterion, ReviewBoard
from .store import Store, content_hash

TERMINAL = ("completed", "failed", "budget_exhausted", "escalated", "blocked")


@dataclass
class Message:
    id: str
    to_node: str
    kind: str                       # instruction|observation|question|cancel|scope_change
    payload: dict[str, Any]
    hops: int = 0
    visited: tuple[str, ...] = ()
    max_hops: int = 8


class MessageBus:
    """Addressed transport; the org tree carries AUTHORITY, not packets.

    Tree-only routing does deliver (simulation: >94% even at 1M nodes) but it
    burns a routing decision per hop -- 7.6 model calls instead of 1 at b=10,
    d=4 -- and every relay hop is a scope-leak opportunity. So the tree decides
    who may commit what; the bus decides who receives what.
    """

    def __init__(self, store: Store):
        self.store = store
        self.inbox: dict[str, list[Message]] = {}
        self.dead_letters: list[Message] = []

    def send(self, run_id: str, sender: str, msg: Message) -> None:
        if msg.hops >= msg.max_hops or msg.to_node in msg.visited:
            self.dead_letters.append(msg)
            self.store.append_event(run_id, sender, "message_dead_lettered",
                                    {"id": msg.id, "to": msg.to_node,
                                     "hops": msg.hops})
            return
        self.inbox.setdefault(msg.to_node, []).append(msg)
        self.store.append_event(run_id, sender, "message_sent",
                                {"id": msg.id, "to": msg.to_node, "kind": msg.kind})

    def take(self, node_key: str) -> list[Message]:
        return self.inbox.pop(node_key, [])


@dataclass
class NodeResult:
    status: str
    value: Any = None
    partial: bool = False
    reason: str = ""
    tokens: int = 0


@dataclass
class RunStats:
    nodes: int = 0
    capability_calls: int = 0
    planner_calls: int = 0
    library_hits: int = 0
    library_misses: int = 0
    reviews: int = 0
    review_rounds: int = 0
    escalations: int = 0
    validator_rejections: int = 0
    dead_end_hits: int = 0
    demoted: int = 0
    declared_ambiguous: int = 0
    replans: int = 0
    tokens: int = 0
    max_depth: int = 0
    fan_outs: list[int] = field(default_factory=list)
    ambiguous: int = 0
    emitted: int = 0

    @property
    def f_measured(self) -> float:
        return self.ambiguous / self.emitted if self.emitted else 0.0

    @property
    def b_measured(self) -> float:
        return sum(self.fan_outs) / len(self.fan_outs) if self.fan_outs else 0.0

    @property
    def m_measured(self) -> float:
        """Offspring mean = E[ambiguous children per decomposition].

        NOT mean(b) * mean(f): b and f are correlated across problems, and the
        product of the means gave the wrong regime for one of the real
        planners measured (subcritical 1.02 vs the correct 0.88).
        """
        return self.ambiguous / len(self.fan_outs) if self.fan_outs else 0.0

    @property
    def m_declared(self) -> float:
        """What the planner CLAIMED, before admission control demoted anything."""
        return (self.declared_ambiguous / len(self.fan_outs)
                if self.fan_outs else 0.0)


class Crash(BaseException):
    """Raised by the crash hook. Deliberately not an Exception subclass so no
    `except Exception` in the executor can accidentally swallow it."""


class Runtime:
    def __init__(
        self,
        store: Store,
        registry: CapabilityRegistry,
        library: SolvedProblemLibrary,
        planner: Planner,
        board: ReviewBoard | None = None,
        max_depth: int = 6,
        review_plans: bool = True,
        session_prefix: str = "sess",
        admission: "Callable[[Step, str], tuple[bool, str]] | None" = None,
    ):
        self.store = store
        self.registry = registry
        self.library = library
        self.planner = planner
        self.board = board or ReviewBoard(store)
        self.max_depth = max_depth
        self.review_plans = review_plans
        self.bus = MessageBus(store)
        self.session_prefix = session_prefix
        self._session_n = 0
        self.stats = RunStats()
        self.crash_at: str | None = None
        # ADMISSION CONTROL. Measured on real planners (probe_optimism.py):
        # 35-44% of steps a planner marks "atomic" cannot in fact be done by
        # the capability it names. Declared `f` therefore understates the
        # branching mean by more than 10x on the best planner tested -- its
        # apparent m of 0.12 was really 1.62. "Atomic" has to be a claim that
        # is checked, not a label the author gets to assign. A step that fails
        # admission is demoted to `decompose` rather than executed.
        self.admission = admission

    # ---------------------------------------------------------------- helpers

    def new_session(self, role: str) -> str:
        self._session_n += 1
        return f"{self.session_prefix}-{role}-{self._session_n}"

    def completed_nodes(self, root_run_id: str) -> dict[str, Any]:
        """Projection: which (run, node) pairs already finished, and with what."""
        done: dict[str, Any] = {}
        for ev in self.store.events(root_run_id):
            if ev["type"] == "node_completed":
                done[f"{ev['run_id']}/{ev['node_id']}"] = json.loads(ev["payload"])
        return done

    def projection(self, root_run_id: str) -> dict[str, Any]:
        """The whole run, rebuilt from events. Two runs are equal iff these are."""
        tree: dict[str, Any] = {}
        results: dict[str, Any] = {}
        statuses: dict[str, str] = {}
        for ev in self.store.events(root_run_id):
            t, payload = ev["type"], json.loads(ev["payload"])
            key = f"{ev['run_id']}/{ev['node_id']}"
            if t == "run_created":
                tree[ev["run_id"]] = {"parent": ev["parent_run_id"],
                                      "problem": payload.get("problem")}
            elif t == "node_completed":
                results[key] = payload.get("value")
                statuses[key] = payload.get("status", "completed")
            elif t in ("node_budget_exhausted", "node_failed", "node_escalated"):
                statuses[key] = t.replace("node_", "")
        return {"tree": tree, "results": results, "statuses": statuses}

    # --------------------------------------------------------------- execution

    def run(
        self,
        problem: str,
        budget: Budget,
        authority: Authority,
        run_id: str = "root",
        signature: str = "any->any",
        resume: bool = False,
    ) -> NodeResult:
        skip = self.completed_nodes(run_id) if resume else {}
        if resume:
            self.store.append_event(run_id, run_id, "run_resumed",
                                    {"completed": len(skip)})
        return self._solve(problem, signature, budget, authority, run_id,
                           parent_run_id=None, depth=0, skip=skip)

    def _solve(self, problem: str, signature: str, budget: Budget,
               authority: Authority, run_id: str, parent_run_id: str | None,
               depth: int, skip: dict[str, Any]) -> NodeResult:
        self.stats.max_depth = max(self.stats.max_depth, depth)
        self.store.append_event(run_id, run_id, "run_created",
                                {"problem": problem, "depth": depth,
                                 "budget": budget.as_dict()},
                                parent_run_id=parent_run_id)
        allowance = self.max_depth - depth
        if depth > self.max_depth:
            return self._escalate(run_id, run_id, "depth cap reached", budget)
        if budget.exhausted():
            return self._exhausted(run_id, run_id, budget, None)

        # 0. Has this already been shown to be out of reach at this allowance?
        #    Re-deriving a known dead end is the most expensive way to learn
        #    nothing (scenario S7).
        dead = self.library.lookup_intractable(problem, signature, allowance)
        if dead is not None:
            self.stats.dead_end_hits += 1
            return self._escalate(
                run_id, run_id,
                f"known dead end after {dead.attempts} attempt(s): {dead.reason}",
                budget)

        # 1. Has this been solved before? (the termination mechanism)
        hit = self.library.lookup(problem, signature)
        if hit is not None:
            self.stats.library_hits += 1
            plan = _plan_from_json(hit.plan_json)
            self.store.append_event(run_id, run_id, "plan_reused",
                                    {"key": hit.key,
                                     "reliability": round(hit.reliability, 3)})
        else:
            self.stats.library_misses += 1
            plan = self._author_plan(problem, signature, authority, run_id,
                                     budget, depth)
            if plan is None:
                return self._escalate(run_id, run_id,
                                      "no valid plan after replanning", budget)

        result = self._execute_plan(plan, budget, authority, run_id, depth, skip)

        if result.status in ("escalated", "budget_exhausted") and depth > 0:
            self.library.record_intractable(problem, signature, result.reason,
                                            allowance)
        if hit is not None:
            self.library.record_use(hit.key, result.status)
        elif result.status == "completed":
            self.library.publish(problem, signature, plan,
                                 {"nodes": self.stats.nodes,
                                  "tokens": budget.spent_tokens},
                                 provenance=run_id)
        return result

    def _author_plan(self, problem: str, signature: str, authority: Authority,
                     run_id: str, budget: Budget, depth: int) -> Plan | None:
        """Plan -> admit -> validate -> review. In that order.

        Admission runs BEFORE the branching statistics are recorded, because
        the whole point of the measurement is that the planner's own count of
        ambiguous steps is not the quantity that governs termination.
        """
        author = self.new_session("planner")
        for attempt in range(3):
            draft = self.planner.decompose(problem, signature,
                                           self.registry.maturities(), depth)
            self.stats.planner_calls += 1
            self.stats.tokens += draft.tokens_in + draft.tokens_out
            budget.spend(tokens=draft.tokens_in + draft.tokens_out)
            declared_ambiguous = draft.n_ambiguous

            if self.admission is not None:
                draft = PlanDraft(
                    Plan(draft.plan.id, draft.plan.problem,
                         tuple(self._admit(st, run_id) for st in draft.plan.steps),
                         draft.plan.output_schema, draft.plan.signature),
                    draft.rationale, draft.tokens_in, draft.tokens_out,
                    draft.raw, draft.model)

            self.stats.fan_outs.append(draft.fan_out)
            self.stats.emitted += draft.fan_out
            self.stats.ambiguous += draft.n_ambiguous
            self.stats.declared_ambiguous += declared_ambiguous
            self.store.append_event(
                run_id, run_id, "plan_drafted",
                {"attempt": attempt, "steps": draft.fan_out,
                 "ambiguous": draft.n_ambiguous,
                 "declared_ambiguous": declared_ambiguous,
                 "demoted": draft.n_ambiguous - declared_ambiguous,
                 "f": round(draft.f, 3), "model": draft.model,
                 "rationale": draft.rationale[:300],
                 "plan": as_jsonable(draft.plan)},
                session_id=author,
            )
            errors = validate(draft.plan, self.registry.maturities(), authority)
            if not errors:
                if self.review_plans:
                    self._review_plan(draft, run_id, author)
                return draft.plan
            self.stats.validator_rejections += 1
            self.stats.replans += 1
            self.store.append_event(run_id, run_id, "plan_rejected",
                                    {"attempt": attempt, "errors": errors},
                                    session_id=author)
        return None

    def _admit(self, step: Step, run_id: str) -> Step:
        """Verify an `atomic` claim before the runtime acts on it."""
        if step.kind != "capability" or self.admission is None:
            return step
        ok, why = self.admission(step, run_id)
        if ok:
            return step
        self.stats.demoted += 1
        self.store.append_event(
            run_id, step.id, "atomicity_rejected",
            {"capability": step.ref, "reason": why,
             "declared_output": step.output_schema})
        return Step(
            id=step.id, kind="decompose",
            problem=f"{why} (was claimed atomic via {step.ref})",
            outputs=("y",), output_schema=step.output_schema or "any",
            authority=step.authority,
        )

    def _review_plan(self, draft: PlanDraft, run_id: str, author: str) -> None:
        """Evidential review of the PLAN, before a single token is spent on it."""
        reviewer = self.new_session("reviewer")
        artifact = ArtifactVersion(f"plan:{run_id}", 1,
                                   json.dumps(as_jsonable(draft.plan),
                                              sort_keys=True), author)
        maturities = self.registry.maturities()

        def bounded(body: str) -> tuple[bool, str]:
            p = json.loads(body)
            bad = [s["id"] for s in p["steps"]
                   if s["kind"] == "loop" and not s.get("max_iterations")]
            return (not bad, f"unbounded loops: {bad}")

        def known(body: str) -> tuple[bool, str]:
            p = json.loads(body)
            bad = [s["ref"] for s in p["steps"]
                   if s["kind"] == "capability" and s["ref"] not in maturities]
            return (not bad, f"unknown capabilities: {bad}")

        def sized(body: str) -> tuple[bool, str]:
            p = json.loads(body)
            return (len(p["steps"]) <= 10, f"{len(p['steps'])} steps")

        criteria = [
            Criterion("C1", "every loop has a static bound", bounded),
            Criterion("C2", "every capability referenced exists", known),
            Criterion("C3", "fan-out is at most 10 steps", sized),
        ]
        _, outcome = self.board.run(
            artifact, criteria, reviewer,
            revise=lambda a, fs: ArtifactVersion(a.artifact_id, a.version + 1,
                                                 a.body, a.author_session),
        )
        self.stats.reviews += 1
        self.stats.review_rounds += outcome.rounds
        self.stats.tokens += outcome.reviewer_tokens

    def _execute_plan(self, plan: Plan, budget: Budget, authority: Authority,
                      run_id: str, depth: int, skip: dict[str, Any]) -> NodeResult:
        last: NodeResult = NodeResult("completed", None)
        partial = False
        unresolved: list[str] = []
        for step in plan.steps:
            key = f"{run_id}/{step.id}"
            if key in skip:
                last = NodeResult(skip[key].get("status", "completed"),
                                  skip[key].get("value"))
                continue
            for msg in self.bus.take(key):
                self.store.append_event(run_id, step.id, "message_delivered",
                                        {"id": msg.id, "kind": msg.kind,
                                         "payload": msg.payload})
                if msg.kind == "cancel":
                    return self._escalate(run_id, step.id, "cancelled by message",
                                          budget)
                if msg.kind == "scope_change" and "args" in msg.payload:
                    step = Step(**{**step.__dict__,
                                   "args": {**step.args, **msg.payload["args"]}})
            if self.crash_at == key:
                self.store.append_event(run_id, step.id, "node_started",
                                        {"crash": True})
                raise Crash(key)
            if budget.exhausted():
                self._exhausted(run_id, step.id, budget, last.value)
                return NodeResult("budget_exhausted", last.value, partial=True,
                                  reason=f"budget spent at step {step.id}")
            self.stats.nodes += 1
            budget.spend(nodes=1)
            self.store.append_event(run_id, step.id, "node_started",
                                    {"kind": step.kind})
            res = self._execute_step(step, budget, authority, run_id, depth, skip)
            partial = partial or res.partial
            self.store.append_event(
                run_id, step.id,
                "node_completed" if res.status == "completed" else f"node_{res.status}",
                {"status": res.status, "value": res.value, "reason": res.reason},
            )
            if res.status == "failed":
                return res
            if res.status == "budget_exhausted":
                return NodeResult("budget_exhausted", last.value, partial=True,
                                  reason=res.reason)
            if res.status == "escalated":
                # FIX (scenario S7): the first version returned here, so one
                # unreachable subtree cancelled every sibling that could still
                # have been solved -- and with them everything the run would
                # have learned. Harvest what is reachable, then escalate once,
                # naming what is unresolved.
                unresolved.append(step.id)
                partial = True
                continue
            last = res
        if unresolved:
            return NodeResult("escalated", last.value, partial=True,
                              reason=f"unresolved steps: {unresolved}")
        return NodeResult("completed", last.value, partial=partial)

    def _execute_step(self, step: Step, budget: Budget, authority: Authority,
                      run_id: str, depth: int, skip: dict[str, Any]) -> NodeResult:
        node_authority = authority.narrow(step.authority) if (
            step.authority != Authority()) else authority
        if step.kind == "capability":
            self.stats.capability_calls += 1
            try:
                value = self.registry.invoke(step.ref, step.args, run_id, step.id,
                                             self.new_session("worker"),
                                             node_authority)
            except (PermissionError, KeyError, RuntimeError) as exc:
                return NodeResult("failed", None, reason=str(exc))
            budget.spend(tokens=200, usd=0.0002)
            return NodeResult("completed", value)
        if step.kind == "decompose":
            child_budget = budget.child_share(1)
            child_run = f"{run_id}/{step.id}"
            res = self._solve(step.problem or "", f"any->{step.output_schema}",
                              child_budget, node_authority, child_run, run_id,
                              depth + 1, skip)
            budget.absorb(child_budget)
            if res.status == "budget_exhausted":
                # P3: the child ran out. The PARENT decides -- top up from the
                # reserve once, then escalate. Never silently accept partial.
                moved = child_budget.top_up(budget, int(budget.remaining().tokens * 0.5))
                self.store.append_event(run_id, step.id, "budget_reallocated",
                                        {"moved_tokens": moved,
                                         "child": child_run})
                self.stats.escalations += 1
                if moved <= 0:
                    return NodeResult("budget_exhausted", res.value, partial=True,
                                      reason="child exhausted, no reserve left")
                res = self._solve(step.problem or "", f"any->{step.output_schema}",
                                  child_budget, node_authority, child_run, run_id,
                                  depth + 1, self.completed_nodes(child_run))
                budget.absorb(child_budget)
            return res
        if step.kind == "branch":
            arm = step.then if _truthy(step.guard) else step.otherwise
            return self._execute_plan(Plan(f"{step.id}-arm", "", tuple(arm)),
                                      budget, node_authority, run_id, depth, skip)
        if step.kind == "loop":
            out: Any = None
            for i in range(step.max_iterations or 0):
                if budget.exhausted():
                    return NodeResult("budget_exhausted", out, partial=True,
                                      reason=f"loop {step.id} at iteration {i}")
                r = self._execute_plan(Plan(f"{step.id}-body-{i}", "",
                                            tuple(step.then)), budget,
                                       node_authority, run_id, depth, skip)
                out = r.value
                if r.status != "completed":
                    return r
            return NodeResult("completed", out)
        if step.kind == "parallel":
            values = []
            for i, branch in enumerate(step.branches):
                r = self._execute_plan(Plan(f"{step.id}-b{i}", "", tuple(branch)),
                                       budget, node_authority, run_id, depth, skip)
                if r.status != "completed":
                    return r
                values.append(r.value)
            return NodeResult("completed", values)
        if step.kind == "return":
            return NodeResult("completed", step.args.get("value"))
        if step.kind == "fail":
            return NodeResult("failed", None, reason=str(step.args.get("reason", "")))
        return NodeResult("failed", None, reason=f"unknown step kind {step.kind!r}")

    # ------------------------------------------------------------- terminals

    def _exhausted(self, run_id: str, node_id: str, budget: Budget,
                   partial: Any) -> NodeResult:
        self.store.append_event(run_id, node_id, "node_budget_exhausted",
                                {"budget": budget.as_dict(), "partial": partial})
        return NodeResult("budget_exhausted", partial, partial=True,
                          reason="budget exhausted")

    def _escalate(self, run_id: str, node_id: str, reason: str,
                  budget: Budget) -> NodeResult:
        self.stats.escalations += 1
        self.store.append_event(run_id, node_id, "node_escalated",
                                {"reason": reason, "budget": budget.as_dict()})
        return NodeResult("escalated", None, partial=True, reason=reason)


def _truthy(guard: str | None) -> bool:
    """Guards are evaluated fail-closed: anything not understood is False."""
    if not guard:
        return False
    return guard.strip().lower() in ("true", "1", "yes")


def _plan_from_json(text: str) -> Plan:
    d = json.loads(text)

    def step(s: dict[str, Any]) -> Step:
        return Step(
            id=s["id"], kind=s["kind"], ref=s.get("ref"), args=s.get("args") or {},
            problem=s.get("problem"), inputs=tuple(s.get("inputs") or ()),
            outputs=tuple(s.get("outputs") or ()), guard=s.get("guard"),
            then=tuple(step(x) for x in s.get("then") or ()),
            otherwise=tuple(step(x) for x in s.get("otherwise") or ()),
            branches=tuple(tuple(step(x) for x in b)
                           for b in s.get("branches") or ()),
            max_iterations=s.get("max_iterations"),
            output_schema=s.get("output_schema", "any"),
        )

    return Plan(d["id"], d["problem"], tuple(step(s) for s in d["steps"]),
                d.get("output_schema", "any"), d.get("signature", "any->any"))

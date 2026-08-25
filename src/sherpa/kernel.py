"""The sherpa kernel: durable execution of typed plans (#492 §2).

Single-machine MVP. All state lives in SQLite (WAL) plus content-addressed
blobs; every side effect is bracketed by events, so killing the process at any
point leaves a resumable log. Resume rebuilds by replay, skips nodes already
completed (their outputs are restored from the log — completed idempotent
effects are never repeated), recovers expired leases, and continues loops from
event-sourced iteration counters.

Terminal states are loud: ``completed``, ``failed``, ``blocked``,
``escalated``, ``cancelled``, ``budget_exhausted``. Partial output can never
look successful: final outputs pass real acceptance checks and an independent
review gate before the run may report ``completed``.

Executor notes: the MVP executor is single-threaded and local (#487 defers
distribution). ``Parallel`` executes its branches to completion even when one
branch fails — an escalating sibling never cancels runnable siblings (measured
finding from #485). The solution cache stores positive results when a child
plan solves its goal; budget exhaustion is never cached as evidence against a
plan (the store refuses that combination outright).
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sherpa.admission import AdmissionChecker
from sherpa.capabilities import (
    CapabilityContext,
    CapabilityRegistry,
    register_builtins,
    resolve_inputs,
    run_capability,
)
from sherpa.channel import ModelChannel, make_channel
from sherpa.context import journal
from sherpa.events import Event
from sherpa.expr import evaluate
from sherpa.ir import (
    TERMINAL_STATES,
    AskUser,
    Authority,
    Branch,
    Budgets,
    Decompose,
    Fail,
    InvokeCapability,
    InvokePlan,
    Parallel,
    Plan,
    ProblemSpec,
    Return,
    While,
    iter_nodes,
    validate_plan,
)
from sherpa.metrics import run_metrics
from sherpa.planner import PlanAuthoringError, Planner, StubPlanner, plan_signature
from sherpa.review import ReviewPolicy, Reviewer
from sherpa.store import Store


FINAL_STATES = frozenset({"completed", "failed", "escalated", "cancelled", "budget_exhausted"})


class RunResult(BaseModel):
    run_id: str
    status: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    workspace: str = ""


class _PlanRefused(Exception):
    """A plan was refused by delegation or by the review gate.

    Previously these were bare ``PermissionError``/``ValueError`` raised out of
    ``_author_child``/``_root_plan``, which escaped ``Engine.run`` and left the
    run stranded in ``running`` with no ``run_terminal`` event.
    """


class _DepthExceeded(Exception):
    """Recursive decomposition would exceed ``budgets.max_depth``.

    Depth was previously a local that was never incremented, so ``max_depth``
    constrained only the structural nesting of branch/while/parallel inside a
    single plan and never bounded recursion at all.
    """


class _BudgetExhausted(Exception):
    pass


def _new_id(prefix: str) -> str:
    return f"{prefix}_{os.urandom(6).hex()}"


class Engine:
    """Public API: run / resume / status / export_trace / deliver_message."""

    def __init__(
        self,
        workspace: Path,
        *,
        channel_policy: str = "recorded",
        recordings: dict[str, list[str]] | None = None,
        registry: CapabilityRegistry | None = None,
        planner: Planner | None = None,
        db_path: Path | None = None,
        fault_injection: bool = False,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        #: Crash-injection is opt-in per Engine. Reading SHERPA_KILL_AFTER_EVENTS
        #: unconditionally meant a stray environment variable could SIGKILL a
        #: production process mid-run.
        self.fault_injection = fault_injection
        self.store = Store(db_path or self.workspace / "sherpa.db")
        self.registry = registry or CapabilityRegistry()
        if not self.registry.names():
            register_builtins(self.registry)
        self.channel: ModelChannel = make_channel(channel_policy, recordings)
        self.planner = planner or StubPlanner(registry_names=self.registry.names())
        self.admission = AdmissionChecker(self.store, self.registry, self.store.blob)

    # ------------------------------------------------------------------ API

    def run(self, problem: ProblemSpec | Path | dict, *, run_id: str | None = None) -> RunResult:
        spec = self._coerce_problem(problem)
        rid = run_id or _new_id("run")
        spec_sha = self.store.blob.put_text(spec.model_dump_json())
        self.store.create_run(rid, problem_sha=spec_sha)
        self.store.append(
            Event(kind="plan_recorded", run_id=rid,
                  payload={"problem": spec.model_dump(), "spec_sha": spec_sha})
        )
        return self._execute(rid, spec)

    def resume(self, run_id: str) -> RunResult:
        """Continue a run after crash/pause/attendee-answer.

        ``blocked`` is loud-but-continuable: it is how attended runs wait for
        input. Only the FINAL states refuse resume.
        """
        proj = self.store.projection(run_id)
        if proj["status"] in FINAL_STATES:
            return self._result(run_id, proj["status"], proj["error"])
        self._recover_orphans(run_id)
        spec = self._problem_from_log(run_id)
        return self._execute(run_id, spec, resumed=True)

    def status(self, run_id: str) -> dict:
        return self.store.projection(run_id)

    def deliver_message(self, run_id: str, to_node_key: str, payload: dict,
                        sender: str = "user") -> int:
        return self.store.enqueue_message(run_id, to_node_key, payload, sender=sender)

    def pause(self, run_id: str) -> None:
        self.store.set_run_status(run_id, "paused")

    def cancel(self, run_id: str) -> None:
        self.store.set_run_status(run_id, "cancelled")

    def export_trace(self, run_id: str, path: Path) -> Path:
        trace = {
            "run_id": run_id,
            "projection": self.store.projection(run_id),
            "replay_projection": self.store.replay_projection(run_id),
            "events": [e.model_dump() for e in self.store.events(run_id=run_id)],
            "metrics": run_metrics(self.store.events(run_id=run_id)),
        }
        Path(path).write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
        return Path(path)

    def close(self) -> None:
        self.store.close()

    # -------------------------------------------------------------- internals

    def _coerce_problem(self, problem: ProblemSpec | Path | dict) -> ProblemSpec:
        if isinstance(problem, ProblemSpec):
            return problem
        if isinstance(problem, Path):
            return ProblemSpec(**json.loads(Path(problem).read_text(encoding="utf-8")))
        if isinstance(problem, dict):
            return ProblemSpec(**problem)
        raise TypeError(f"cannot coerce {type(problem)!r} to ProblemSpec")

    def _problem_from_log(self, run_id: str) -> ProblemSpec:
        for e in self.store.events(run_id=run_id, kinds=["plan_recorded"]):
            return ProblemSpec(**e.payload["problem"])
        raise KeyError(f"run {run_id} has no recorded problem")

    def _recover_orphans(self, run_id: str) -> None:
        for r, node_key, owner in [t for t in self.store.expired_leases() if t[0] == run_id]:
            self.store.release_lease(r, node_key, owner)
            state = self.store.projection_node_state(r, node_key)
            if state == "leased":
                self.store.cas_node_state(r, node_key, "leased", "pending")
            self.store.append(Event(kind="orphan_recovered", run_id=r, node_key=node_key,
                                    payload={"former_owner": owner}))

    def _maybe_kill(self) -> None:
        """Fault-injection hook: REAL SIGKILL once the log reaches N events.

        Inert unless this Engine was constructed with ``fault_injection=True``.
        """
        if not self.fault_injection:
            return
        after = os.environ.get("SHERPA_KILL_AFTER_EVENTS")
        if after and self.store.head_seq() >= int(after):
            os.kill(os.getpid(), signal.SIGKILL)

    def _run_started_ts(self, run_id: str) -> float:
        for e in self.store.events(run_id=run_id, limit=1):
            return e.ts
        return time.time()

    def _check_budgets(self, rid: str, spec: ProblemSpec) -> None:
        u = self.store.usage(rid)
        b = spec.budgets
        wall = time.time() - self._run_started_ts(rid)
        over = (
            u["tokens"] > b.max_tokens
            or u["cost_usd"] > b.max_cost_usd
            or u["nodes"] > b.max_nodes
            or wall > b.max_wall_seconds
        )
        if over:
            raise _BudgetExhausted()

    def _check_depth(self, spec: ProblemSpec, next_depth: int) -> None:
        if next_depth > spec.budgets.max_depth:
            raise _DepthExceeded(
                f"decomposition depth {next_depth} exceeds max_depth "
                f"{spec.budgets.max_depth}"
            )

    def _terminal(self, rid: str, status: str, error: str | None = None) -> RunResult:
        self.store.set_run_status(rid, status, error)
        self._maybe_kill()
        return self._result(rid, status, error)

    def _result(self, rid: str, status: str, error: str | None = None) -> RunResult:
        outputs: dict[str, Any] = {}
        for e in self.store.events(run_id=rid, kinds=["node_state_changed"]):
            ret = e.payload.get("return_outputs")
            if isinstance(ret, dict):
                outputs = ret
        m = run_metrics(self.store.events(run_id=rid))
        m["run_id"] = rid
        return RunResult(run_id=rid, status=status, outputs=outputs, error=error,
                         metrics=m, workspace=str(self.workspace))

    def _execute(self, rid: str, spec: ProblemSpec, *, resumed: bool = False) -> RunResult:
        try:
            return self._drive(rid, spec, resumed=resumed)
        except _BudgetExhausted:
            journal(self.store, rid, None, "blocker",
                    "budget exhausted; stopping loudly", refs=["budgets"])
            return self._terminal(rid, "budget_exhausted")
        except _DepthExceeded as exc:
            journal(self.store, rid, None, "blocker", str(exc), refs=["budgets"])
            return self._terminal(rid, "budget_exhausted", error=str(exc))
        except _PlanRefused as exc:
            journal(self.store, rid, None, "blocker", str(exc), refs=["review"])
            return self._terminal(rid, "escalated", error=str(exc))
        except PlanAuthoringError as exc:
            # A planner that cannot author a plan is a loud, recorded outcome.
            # This used to propagate out of Engine.run, leaving the run stranded
            # in `running` with no run_terminal event and the node still pending.
            reason = f"planner could not author a plan: {type(exc).__name__}: {exc}"
            journal(self.store, rid, None, "blocker", reason, refs=["planner"])
            return self._terminal(rid, "escalated", error=reason)

    # ------------------------------------------------------------- main loop

    def _drive(self, rid: str, spec: ProblemSpec, *, resumed: bool) -> RunResult:
        session = f"worker_{rid[-6:]}"
        scope: dict[str, Any] = {"inputs": spec.inputs}
        root_plan = self._root_plan(rid, spec, author_session=f"planner_{rid[-6:]}")
        ctx_cache: dict[tuple[str, str], CapabilityContext] = {}
        pending_decompose: dict[str, tuple[str, str]] = {}
        depth = 0

        # (plan, nodes, index, epoch, depth, parent_key). `depth` used to be a
        # local initialised to 0 and never incremented, and `parent_key` was
        # held only in a transient dict, so neither survived the process.
        stack: list[tuple[Plan, list, int, int, int, str | None]] = [
            (root_plan, list(root_plan.root), 0, 0, 0, None)
        ]
        while stack:
            self._check_budgets(rid, spec)
            plan, nodes, idx, epoch, depth, parent_key = stack.pop()
            if idx >= len(nodes):
                child_sig = pending_decompose.pop(plan.id, None)
                if child_sig is not None:
                    sig, parent_key = child_sig
                    parent_state = self.store.projection_node_state(rid, parent_key)
                    reclassified = parent_state == "skipped"
                    self.store.cache_put(sig, {"status_class": "solved",
                                               "plan": plan.model_dump()})
                    self.store.append(Event(
                        kind="decompose_outcome", run_id=rid, node_key=parent_key,
                        payload={**self._decompose_stats(plan),
                                 "reclassified": reclassified,
                                 "parent_state": parent_state}))
                    if parent_state == "pending":
                        self.store.cas_node_state(rid, parent_key, "pending", "completed")
                continue

            node = nodes[idx]
            node_key = self._node_key(plan, node, epoch)
            stack.append((plan, nodes, idx + 1, epoch, depth, parent_key))
            state = self.store.projection_node_state(rid, node_key)

            if isinstance(node, Return):
                outs = resolve_inputs(node.outputs, scope)
                self.store.append(Event(kind="node_state_changed", run_id=rid, node_key=node_key,
                                        payload={"new": "completed", "return_outputs": outs}))
                if plan.id != root_plan.id:
                    # Child-plan Return: hand outputs to the parent scope and
                    # let the parent continue; only the ROOT Return ends the run.
                    scope[f"_outputs_{plan.id}"] = outs
                    journal(self.store, rid, node_key, "result",
                            f"child plan {plan.id} returned {sorted(outs)}")
                    continue
                verdict = self._final_review(rid, spec, outs, author_session=session)
                if verdict.verdict == "blocked_escalated":
                    blocking = [f.model_dump() for f in verdict.findings if f.blocking]
                    return self._terminal(rid, "escalated",
                                          error=f"final review blocked: {blocking}")
                journal(self.store, rid, node_key, "result",
                        f"outputs accepted; residual risks: {len(verdict.residual_risks)}")
                return self._terminal(rid, "completed")

            if isinstance(node, InvokeCapability):
                if state == "completed":
                    self._restore_node_outputs(rid, node_key, scope)
                    continue
                if not self._begin_attempt(rid, node_key, session, depth, parent_key):
                    continue
                msgs = self.store.take_messages(rid, node_key)
                if msgs:
                    scope[f"{node.id}_msg"] = msgs[-1]
                    journal(self.store, rid, node_key, "decision",
                            f"scope-change message consumed at checkpoint: {msgs[-1]}")
                resolved = resolve_inputs(node.inputs, scope)
                ctx = self._ctx(rid, node_key, ctx_cache, spec.authority)
                verdict = self.admission.check(node, resolved, spec.authority, ctx)
                if verdict.decision == "admitted":
                    try:
                        cap = self.registry.get(node.capability)
                        result = run_capability(cap, resolved, ctx, spec.authority)
                        scope[node.id] = {"result": result}
                        self.store.add_usage(rid, attempts=1, nodes=1)
                        self.store.cas_node_state(rid, node_key, "running", "completed")
                        self.store.release_lease(rid, node_key, session)
                    except Exception as exc:  # noqa: BLE001 - journaled loud failure
                        reason = f"{type(exc).__name__}: {exc}"
                        journal(self.store, rid, node_key, "blocker", reason)
                        self.store.cas_node_state(rid, node_key, "running", "failed")
                        self.store.append(Event(kind="attempt_finished", run_id=rid,
                                                node_key=node_key,
                                                payload={"ok": False, "error": reason}))
                        return self._fail_fast(rid, reason)
                elif verdict.decision == "reclassify_decompose":
                    self.store.add_usage(rid, attempts=1)
                    goal = f"achieve {node.capability} without a direct call"
                    hints = {
                        "requested_capability": node.capability,
                        "original_step": node.model_dump(),
                        "admission_reasons": verdict.reasons,
                    }
                    self._check_depth(spec, depth + 1)
                    child = self._author_child(rid, spec, goal, hints,
                                               spec.authority, spec.budgets, depth + 1)
                    sig = plan_signature(goal, hints, spec.authority, spec.budgets)
                    self.store.cas_node_state(rid, node_key, "running", "skipped")
                    pending_decompose[child.id] = (sig, node_key)
                    stack.append((child, list(child.root), 0, 0, depth + 1, node_key))
                else:
                    journal(self.store, rid, node_key, "blocker",
                            "; ".join(verdict.reasons))
                    self.store.cas_node_state(rid, node_key, "running", "escalated")
                    self.store.release_lease(rid, node_key, session)
                    return self._terminal(rid, "escalated", error="; ".join(verdict.reasons))
                self._maybe_kill()
                continue

            if isinstance(node, Decompose):
                if state == "completed":
                    continue
                self._ensure_node(rid, node_key, depth, parent_key)
                hints = {**node.hints,
                         "prior_results": {k: v.get("result") for k, v in scope.items()
                                           if isinstance(v, dict) and "result" in v}}
                self._check_depth(spec, depth + 1)
                sig = plan_signature(node.subgoal, hints, spec.authority, spec.budgets)
                cached = self.store.cache_get(sig)
                if cached and cached.get("status_class") == "solved" and cached.get("plan"):
                    self.store.append(Event(kind="cache_hit", run_id=rid, node_key=node_key,
                                            payload={"signature": sig}))
                    child = Plan(**cached["plan"])
                else:
                    child = self._author_child(rid, spec, node.subgoal, hints,
                                               spec.authority, spec.budgets, depth + 1)
                pending_decompose[child.id] = (sig, node_key)
                stack.append((child, list(child.root), 0, 0, depth + 1, node_key))
                continue

            if isinstance(node, Branch):
                chosen = next(
                    (c for c in node.cases if c.when is None or evaluate(c.when, scope)), None
                )
                if chosen is not None:
                    stack.append((plan, list(chosen.body), 0, epoch, depth, parent_key))
                continue

            if isinstance(node, While):
                iters = self._loop_iterations(rid, node_key)
                if iters < node.max_iterations and evaluate(node.guard, scope):
                    self.store.append(Event(kind="node_progress", run_id=rid,
                                            node_key=node_key,
                                            payload={"iterations": iters + 1}))
                    next_epoch = iters + 1
                    stack.append((plan, [node], 0, epoch, depth, parent_key))
                    stack.append((plan, list(node.body), 0, next_epoch, depth, parent_key))
                continue

            if isinstance(node, Parallel):
                for branch in reversed(node.branches):
                    stack.append((plan, list(branch), 0, epoch, depth, parent_key))
                continue

            if isinstance(node, AskUser):
                if state == "completed":
                    continue
                self._ensure_node(rid, node_key, depth, parent_key)
                if spec.attended and depth == 0:
                    msgs = self.store.take_messages(rid, node_key)
                    if not msgs:
                        self.store.cas_node_state(rid, node_key, "pending", "blocked")
                        return self._terminal(rid, "blocked",
                                              error=(f"awaiting input at {node_key}: "
                                                     f"{node.question}"))
                    scope[node.id] = {"answer": msgs[-1]}
                    self.store.cas_node_state(rid, node_key, "blocked", "completed")
                else:
                    self.store.cas_node_state(rid, node_key, "pending", "escalated")
                    return self._terminal(rid, "escalated",
                                          error=f"ask_user outside attended root: {node_key}")
                continue

            if isinstance(node, Fail):
                self._ensure_node(rid, node_key, depth, parent_key)
                self.store.cas_node_state(rid, node_key, "pending", "failed")
                return self._terminal(rid, "failed", error=node.reason)

            if isinstance(node, InvokePlan):
                raise NotImplementedError("invoke_plan binds the solution library (#489)")

        return self._terminal(rid, "failed", error="plan exhausted without Return")

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _root_plan_id(spec: ProblemSpec) -> str:
        return f"root_{spec.id}"

    @staticmethod
    def _node_key(plan: Plan, node: Any, epoch: int) -> str:
        base = f"{plan.id}.{node.id}"
        return f"{base}@{epoch}" if epoch else base

    def _fail_fast(self, rid: str, reason: str) -> RunResult:
        journal(self.store, rid, None, "blocker", f"fail-fast: {reason}", refs=["kernel"])
        return self._terminal(rid, "failed", error=reason)

    def _loop_iterations(self, rid: str, node_key: str) -> int:
        n = 0
        for e in self.store.events(run_id=rid, kinds=["node_progress"]):
            if e.node_key == node_key:
                n = max(n, int(e.payload.get("iterations", 0)))
        return n

    def _decompose_stats(self, child: Plan) -> dict:
        caps = [n for n in iter_nodes(child.root) if isinstance(n, InvokeCapability)]
        ambiguous = sum(1 for c in caps if not c.atomic_claim)
        return {"children_declared": len(caps) or len(child.root),
                "children_ambiguous": ambiguous}

    def _ensure_node(self, rid: str, node_key: str, depth: int,
                     parent_key: str | None = None) -> None:
        if self.store.projection_node_state(rid, node_key) is None:
            self.store.upsert_node(rid, node_key, "pending", depth=depth,
                                   parent_key=parent_key)

    def _begin_attempt(self, rid: str, node_key: str, session: str, depth: int,
                       parent_key: str | None = None) -> bool:
        self._ensure_node(rid, node_key, depth, parent_key)
        state = self.store.projection_node_state(rid, node_key)
        if state in ("failed", "cancelled", "escalated", "completed"):
            return False
        if not self.store.acquire_lease(rid, node_key, session):
            # Another live session holds this node. The lease answered
            # correctly; honouring it is what makes execution exactly-once.
            self.store.append(Event(kind="lease_denied", run_id=rid, node_key=node_key,
                                    payload={"session": session}))
            return False
        self.store.cas_node_state(rid, node_key, state, "running", owner_session=session)
        self.store.append(Event(kind="attempt_started", run_id=rid, node_key=node_key,
                                payload={"session": session}))
        return True

    def _restore_node_outputs(self, rid: str, node_key: str, scope: dict) -> None:
        for e in reversed(self.store.events(run_id=rid, kinds=["tool_call_finished"])):
            if e.node_key != node_key or not e.payload.get("ok"):
                continue
            sha = e.payload.get("output_sha")
            nid = node_key.rsplit(".", 1)[-1]
            if sha and self.store.blob.exists(sha):
                raw = self.store.blob.get_text(sha)
                try:
                    scope[nid] = {"result": json.loads(raw)}
                except json.JSONDecodeError:
                    scope[nid] = {"result": raw}
            else:
                scope[nid] = {}
            return

    def _root_plan(self, rid: str, spec: ProblemSpec, *, author_session: str) -> Plan:
        meta = spec.metadata or {}
        root_nodes = meta.get("root_nodes")
        if root_nodes:
            plan = Plan(id=self._root_plan_id(spec), authority=spec.authority,
                        budgets=spec.budgets, root=root_nodes)
        else:
            solve: dict[str, Any] = {"kind": "decompose", "id": "solve",
                                     "subgoal": spec.goal, "hints": dict(meta),
                                     "budget_fraction": 1.0}
            plan = Plan(id=self._root_plan_id(spec), authority=spec.authority,
                        budgets=spec.budgets, root=[solve])
        errs = validate_plan(plan)
        if errs:
            detail = ", ".join(f"{e.code}@{e.path}" for e in errs)
            raise ValueError(f"root plan invalid: {detail}")
        report = self._review_gate(rid, spec, plan, author_session)
        if report.verdict == "blocked_escalated":
            blocking = [f.model_dump() for f in report.findings if f.blocking]
            raise _PlanRefused(f"root plan review blocked: {blocking}")
        return plan

    def _review_gate(self, rid: str, spec: ProblemSpec, plan: Plan, author_session: str):
        reviewer = Reviewer(self.store, self.store.blob, lambda: self.channel,
                            policy=ReviewPolicy(max_rounds=1), run_id=rid)
        report = reviewer.review_plan(spec, plan, author_session=author_session)
        journal(self.store, rid, None, "decision",
                f"plan review of {plan.id}@{plan.version}: {report.verdict}")
        if report.verdict == "escalated_review_incomplete":
            # The deterministic checks ran, but the independent model review
            # could not. Recorded as a blocker-level journal entry (and a
            # deferred finding by the reviewer) so the gap is never silent.
            journal(self.store, rid, None, "blocker",
                    f"plan review incomplete for {plan.id}@{plan.version}: "
                    f"{report.channel_error}", refs=["review"])
        return report

    def _author_child(self, rid: str, spec: ProblemSpec, goal: str, hints: dict,
                      granted: Authority, budgets: Budgets, depth: int) -> Plan:
        plan = self.planner.author_plan(goal, hints, granted, budgets,
                                        session=f"planner_{rid[-6:]}")
        if not granted.allows(plan.authority):
            raise _PlanRefused("authored child plan exceeds delegated authority")
        self.store.append(Event(kind="plan_recorded", run_id=rid,
                                payload={"child_plan": plan.model_dump(), "goal": goal}))
        report = self._review_gate(rid, spec, plan, author_session=f"planner_{rid[-6:]}")
        if report.verdict == "blocked_escalated":
            blocking = [f.model_dump() for f in report.findings if f.blocking]
            raise _PlanRefused(f"child plan review blocked: {blocking}")
        return plan

    def _ctx(self, rid: str, node_key: str, cache: dict,
             granted: Authority) -> CapabilityContext:
        key = (rid, node_key)
        if key not in cache:
            cache[key] = CapabilityContext(
                workspace=self.workspace, store=self.store, run_id=rid, node_key=node_key,
                channel_factory=lambda: self.channel, granted=granted,
            )
        return cache[key]

    # ------------------------------------------------------------ final gate

    def _final_review(self, rid: str, spec: ProblemSpec, outputs: dict, *,
                      author_session: str):
        results = self._run_acceptance(spec, outputs)
        reviewer = Reviewer(self.store, self.store.blob, lambda: self.channel,
                            policy=ReviewPolicy(max_rounds=1))
        return reviewer.review_output(spec, {"outputs_json": json.dumps(outputs)},
                                      results, author_session=author_session)

    def _run_acceptance(self, spec: ProblemSpec, outputs: dict) -> dict[str, bool]:
        import subprocess

        results: dict[str, bool] = {}
        out_file = self.workspace / "sherpa_outputs.json"
        out_file.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        for check in spec.acceptance:
            if check.kind == "pytest":
                cmd = list(check.spec.get("cmd", ["pytest", "-q"]))
                argv = [sys.executable, "-m", *cmd]
                cwd = self.workspace / check.spec.get("cwd", ".")
                try:
                    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                                          timeout=300)
                    results[check.id] = proc.returncode == 0
                except Exception:  # noqa: BLE001 - harness failure fails the check
                    results[check.id] = False
            elif check.kind == "predicate":
                try:
                    results[check.id] = bool(evaluate(check.spec["expr"],
                                                      {"outputs": outputs}))
                except Exception:  # noqa: BLE001 - unevaluable predicate fails closed
                    results[check.id] = False
            elif check.kind == "jsonschema":
                from sherpa.admission import check_io

                ok, _ = check_io(outputs, spec.output_schema)
                results[check.id] = ok
        return results

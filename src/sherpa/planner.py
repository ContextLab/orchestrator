"""Planners author typed plan IR (#492 §1, §3).

A planner must be a deterministic function of its inputs — that is what makes
the solution cache meaningful (measured finding from #485). ``StubPlanner``
selects from a declarative plan library; ``LLMPlanner`` authors JSON IR over a
model channel with one repair round, validating against the same schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import ValidationError

from sherpa.ir import Authority, Budgets, Plan, validate_plan

if TYPE_CHECKING:
    from sherpa.channel import ModelChannel


class PlanAuthoringError(Exception):
    """The planner could not produce a valid plan."""


class NoPlanTemplate(PlanAuthoringError):
    """StubPlanner found no matching template in the plan library."""


def plan_signature(goal: str, hints: dict, granted: Authority, budgets: Budgets) -> str:
    """Deterministic cache key over planner inputs."""
    import hashlib

    canonical = {
        "goal": goal,
        "hints": hints,
        "authority": granted.model_dump(),
        "budgets": budgets.model_dump(),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sig_" + hashlib.sha256(blob).hexdigest()[:24]


@runtime_checkable
class Planner(Protocol):
    def author_plan(
        self, goal: str, hints: dict, granted: Authority, budgets: Budgets, session: str
    ) -> Plan: ...


class StubPlanner:
    """Rule-based selection from hints["plan_library"].

    Each entry: ``{"match": {"capability": str | "pattern_contains": str},
    "plan": <Plan dict>}``. First match wins; the mapping is pure.
    """

    def __init__(self, registry_names: set[str] | None = None) -> None:
        self.registry_names = registry_names

    def author_plan(
        self, goal: str, hints: dict, granted: Authority, budgets: Budgets, session: str
    ) -> Plan:
        library = list(hints.get("plan_library", []))
        requested = hints.get("requested_capability")
        for entry in library:
            match = entry.get("match", {})
            if requested is not None and match.get("capability") == requested:
                return self._build(entry["plan"], goal)
            pattern = match.get("pattern_contains")
            if pattern is not None and pattern in goal:
                return self._build(entry["plan"], goal)
        raise NoPlanTemplate(f"no plan-library entry matches goal {goal!r}")

    def _build(self, plan_dict: dict, goal: str) -> Plan:
        plan_dict = dict(plan_dict)
        plan_dict.setdefault("notes", {})
        plan_dict["notes"] = {**plan_dict["notes"], "authored_by": "stub", "goal": goal}
        try:
            plan = Plan(**plan_dict)
        except ValidationError as exc:
            raise PlanAuthoringError(f"library produced invalid plan: {exc}") from exc
        errs = validate_plan(plan, registry_names=self.registry_names)
        if errs:
            raise PlanAuthoringError(f"library plan invalid: {errs}")
        return plan


class LLMPlanner:
    """Model-authored IR validated against the IR schema; one repair round."""

    def __init__(self, channel: "ModelChannel", registry_names: set[str] | None = None) -> None:
        self.channel = channel
        self.registry_names = registry_names
        self.calls = 0

    def _system(self) -> str:
        schema = json.dumps(Plan.model_json_schema(), separators=(",", ":"))
        return (
            "You author plans as strict JSON matching this schema:\n"
            f"{schema}\n"
            "Rules: ids match ^[a-z][a-z0-9_]*$; only the listed node kinds; "
            "no goto; fan-out within budgets; respond with JSON only."
        )

    def author_plan(
        self, goal: str, hints: dict, granted: Authority, budgets: Budgets, session: str
    ) -> Plan:
        user = json.dumps({"goal": goal, "hints": hints, "authority": granted.model_dump(), "budgets": budgets.model_dump()})
        messages = [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for attempt in range(2):
            resp = self.channel.complete(messages, session=session, temperature=0.0, max_tokens=2048)
            self.calls += 1
            try:
                plan = self._parse(resp.text)
                if not granted.allows(plan.authority):
                    raise PlanAuthoringError("plan exceeds delegated authority")
                errs = validate_plan(plan, registry_names=self.registry_names)
                if errs:
                    raise PlanAuthoringError(f"invalid plan: {errs}")
                plan.notes = {**plan.notes, "authored_by": "llm"}
                return plan
            except (PlanAuthoringError, ValidationError) as exc:
                last_error = exc
                messages = messages[:2] + [
                    {"role": "assistant", "content": resp.text},
                    {
                        "role": "user",
                        "content": f"That was not valid ({exc}). Respond again with corrected strict JSON only.",
                    },
                ]
        raise PlanAuthoringError(f"planner failed after repair round: {last_error}")

    def _parse(self, text: str) -> Plan:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end == -1:
            raise PlanAuthoringError("response contained no JSON object")
        data = json.loads(stripped[start : end + 1])
        return Plan(**data)

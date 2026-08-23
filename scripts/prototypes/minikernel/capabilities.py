"""One lifecycle for tools, skills and reusable plans.

#485 keeps tools, skills and pipelines apart, but they need the same things:
an immutable version, a declared contract, a declared authority envelope, a
test suite, an independent qualification, and a complete call history. So this
module gives them one type and one maturity ladder:

    draft -> candidate -> trusted            (promotion, by an independent session)
          -> quarantined -> revoked          (demotion, by an independent session)

Two rules that #485 leaves implicit and that the runtime enforces:

  R1  A capability is never mutated after review. A change publishes a NEW
      version; old runs stay replayable against the version they used.
  R2  The session that authored a capability may never be the session that
      qualifies it -- and qualification means *running* it, not reading it.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .ir import Authority
from .store import Store, content_hash

MATURITIES = ("draft", "candidate", "trusted", "quarantined", "revoked")


@dataclass
class Capability:
    name: str
    version: int
    impl: Callable[[dict[str, Any]], Any]
    input_schema: str
    output_schema: str
    side_effect: str = "pure"          # pure | read | write | external
    authority: Authority = field(default_factory=Authority)
    tests: list[tuple[dict[str, Any], Any]] = field(default_factory=list)
    maturity: str = "draft"
    author_session: str = "unknown"
    source: str = ""

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def content_hash(self) -> str:
        return content_hash(
            f"{self.name}|{self.version}|{self.input_schema}|{self.output_schema}"
            f"|{self.source}"
        )


class SeparationOfDuty(Exception):
    pass


class CapabilityRegistry:
    def __init__(self, store: Store):
        self.store = store
        self._caps: dict[str, Capability] = {}

    # -------------------------------------------------------------- lifecycle

    def register(self, cap: Capability) -> str:
        if cap.ref in self._caps:
            raise ValueError(f"{cap.ref} already registered; publish a new version")
        self._caps[cap.ref] = cap
        self.store.append_event(
            "registry", cap.ref, "capability_registered",
            {"ref": cap.ref, "hash": cap.content_hash, "maturity": cap.maturity,
             "side_effect": cap.side_effect, "author": cap.author_session},
            session_id=cap.author_session,
        )
        return cap.ref

    def seed(self, cap: Capability) -> str:
        """Register a capability that ships with the kernel as already trusted."""
        cap.maturity = "trusted"
        cap.author_session = "kernel"
        return self.register(cap)

    def qualify(self, ref: str, reviewer_session: str) -> tuple[bool, list[str]]:
        """Run the capability's tests in a session that did not author it (R2)."""
        cap = self._caps[ref]
        if reviewer_session == cap.author_session:
            raise SeparationOfDuty(
                f"session {reviewer_session!r} authored {ref} and may not qualify it"
            )
        if not cap.tests:
            self._set_maturity(ref, "quarantined", reviewer_session,
                               "no tests supplied")
            return False, ["capability ships no tests; cannot be qualified"]
        failures: list[str] = []
        for i, (args, expected) in enumerate(cap.tests):
            try:
                got = cap.impl(dict(args))
            except Exception:
                failures.append(f"test {i}: raised\n{traceback.format_exc(limit=2)}")
                continue
            if got != expected:
                failures.append(f"test {i}: expected {expected!r}, observed {got!r}")
        ok = not failures
        self._set_maturity(ref, "trusted" if ok else "quarantined",
                           reviewer_session, "; ".join(failures)[:400])
        return ok, failures

    def revoke(self, ref: str, reviewer_session: str, reason: str) -> None:
        self._set_maturity(ref, "revoked", reviewer_session, reason)

    def _set_maturity(self, ref: str, maturity: str, session: str, reason: str) -> None:
        assert maturity in MATURITIES
        self._caps[ref].maturity = maturity
        self.store.append_event(
            "registry", ref, "capability_maturity_changed",
            {"ref": ref, "maturity": maturity, "reason": reason},
            session_id=session,
        )

    # ------------------------------------------------------------ invocation

    def maturities(self) -> dict[str, str]:
        return {ref: c.maturity for ref, c in self._caps.items()}

    def get(self, ref: str) -> Capability:
        return self._caps[ref]

    def invoke(
        self,
        ref: str,
        args: dict[str, Any],
        run_id: str,
        node_id: str,
        session_id: str,
        authority: Authority,
    ) -> Any:
        cap = self._caps.get(ref)
        if cap is None:
            raise KeyError(f"no such capability {ref!r}")
        if cap.maturity not in ("trusted", "candidate"):
            raise PermissionError(f"{ref} is {cap.maturity}; refusing to invoke")
        if not authority.covers(cap.authority):
            raise PermissionError(
                f"{ref} needs authority the caller does not hold: "
                f"net={sorted(cap.authority.net)} subprocess={cap.authority.subprocess}"
            )
        t0 = time.time()
        try:
            result = cap.impl(dict(args))
            error = None
        except Exception as exc:  # recorded, then re-raised: never swallowed
            result, error = None, f"{type(exc).__name__}: {exc}"
        self.store.append_event(
            run_id, node_id, "capability_invoked",
            {"capability": ref, "hash": cap.content_hash, "args": args,
             "error": error, "seconds": round(time.time() - t0, 4),
             "side_effect": cap.side_effect},
            session_id=session_id,
        )
        if error:
            raise RuntimeError(f"{ref} failed: {error}")
        return result


# ------------------------------------------------------- bug-report workflow


@dataclass
class BugReport:
    id: str
    capability_ref: str
    use_case: str
    expected: str
    observed: str
    reporter_session: str
    status: str = "open"


class BugTracker:
    def __init__(self, store: Store, registry: CapabilityRegistry):
        self.store = store
        self.registry = registry
        self.reports: dict[str, BugReport] = {}

    def file(self, report: BugReport) -> str:
        self.reports[report.id] = report
        self.store.append_event(
            "registry", report.capability_ref, "bug_reported",
            {"id": report.id, "expected": report.expected,
             "observed": report.observed, "use_case": report.use_case},
            session_id=report.reporter_session,
        )
        return report.id

    def triage(self, report_id: str, reviewer_session: str) -> str:
        """A DIFFERENT agent decides, using the call history as evidence."""
        rep = self.reports[report_id]
        if reviewer_session == rep.reporter_session:
            raise SeparationOfDuty("bug reports are triaged by a different session")
        history = self.store.view_tool_history(rep.capability_ref)
        failures = [h for h in history if '"error": null' not in h["payload"]]
        rate = len(failures) / len(history) if history else 0.0
        if rate >= 0.5 and len(history) >= 4:
            self.registry.revoke(rep.capability_ref, reviewer_session,
                                 f"{len(failures)}/{len(history)} invocations failed")
            decision = "revoked"
        elif failures:
            decision = "fix"
        else:
            decision = "clarify_usage"
        rep.status = decision
        self.store.append_event(
            "registry", rep.capability_ref, "bug_triaged",
            {"id": report_id, "decision": decision, "failure_rate": round(rate, 3),
             "invocations": len(history)},
            session_id=reviewer_session,
        )
        return decision

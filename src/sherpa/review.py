"""Bounded independent review (#492 §4).

Review applies to generated plans and final outputs. The reviewer holds a
session distinct from the author's (enforced). Concerns come from a ledger
frozen at review start (contract checks + fixed generic concerns; the hash is
recorded). A finding blocks only with a criterion plus reproducible evidence;
unevidenced reviewer concerns are downgraded to residual risks. Findings have
stable identity and dispositions; rounds/tokens/time are capped; unresolved
blocking findings escalate. A pass verdict is ``pass_with_risk`` — never a
bare "clean".
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field as dfield
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from sherpa.channel import ChannelRequired, ProviderUnavailable, RecordingExhausted

if TYPE_CHECKING:
    from sherpa.channel import ModelChannel
    from sherpa.ir import Plan, ProblemSpec
    from sherpa.store import BlobStore, Store

DISPOSITIONS = ("open", "fixed", "accepted_risk", "invalid", "deferred", "superseded")

#: Model-boundary failures. A review that needed a model and did not get one is
#: incomplete: it escalates with the cause recorded, and never reports a pass.
CHANNEL_FAILURES = (ChannelRequired, RecordingExhausted, ProviderUnavailable)

VERDICT_PASS = "pass_with_risk"
VERDICT_BLOCKED = "blocked_escalated"
VERDICT_INCOMPLETE = "escalated_review_incomplete"


class SeparationOfDutyError(Exception):
    """Reviewer session equals author session."""


def _finding_id(criterion: str, subject: str, evidence_ref: str) -> str:
    blob = json.dumps([criterion, subject, evidence_ref]).encode("utf-8")
    return "find_" + hashlib.sha256(blob).hexdigest()[:20]


class Concern(BaseModel):
    id: str
    criterion: str
    source: str = "generic"


_GENERIC_CONCERNS = (
    ("outputs_conform_to_schema", "final outputs conform to the problem's declared output schema"),
    ("no_acceptance_check_skipped", "every acceptance check in the contract was executed"),
    ("budgets_respected", "recorded usage stays within the delegated budgets"),
)


def concern_ledger(problem: "ProblemSpec") -> list[Concern]:
    """Frozen at review start; identical problems yield identical ledgers."""
    concerns = [Concern(id=f"concern_{c.id}", criterion=c.id, source="contract") for c in problem.acceptance]
    concerns.extend(Concern(id=f"concern_{cid}", criterion=crit, source="generic") for cid, crit in _GENERIC_CONCERNS)
    return concerns


def ledger_sha(concerns: list[Concern]) -> str:
    blob = json.dumps(sorted(c.id for c in concerns)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def ledger_criteria(concerns: list[Concern]) -> set[str]:
    """Criterion keys a finding may cite. Anything else is scope drift."""
    keys: set[str] = set()
    for c in concerns:
        keys.add(c.id)
        keys.add(c.id.removeprefix("concern_"))
    return keys


def ledger_prompt(concerns: list[Concern], sha: str) -> str:
    """The frozen ledger, verbatim, as the reviewer's entire scope for every round."""
    items = "; ".join(f"{c.id} ({c.source}): {c.criterion}" for c in concerns)
    allowed = ", ".join(sorted(c.id.removeprefix("concern_") for c in concerns))
    return (
        f"FROZEN CONCERN LEDGER sha={sha}. These concerns, and only these, are in scope "
        f"for every round of this review: {items}. Each finding's \"criterion\" MUST be one "
        f"of: {allowed}. A finding citing anything else is out of scope and cannot block."
    )


class Finding(BaseModel):
    id: str
    criterion: str
    subject: str
    evidence_ref: str = ""
    blocking: bool = False
    disposition: str = "open"
    rationale: str = ""


class ReviewPolicy(BaseModel):
    max_rounds: int = 2
    max_tokens: int = 20_000
    max_seconds: float = 120.0


class ReviewReport(BaseModel):
    verdict: str
    findings: list[Finding] = Field(default_factory=list)
    residual_risks: list[Finding] = Field(default_factory=list)
    rounds: int = 0
    tokens: int = 0
    ledger_sha: str = ""
    channel_error: str = ""


@dataclass
class Reviewer:
    store: "Store"
    blob: "BlobStore"
    channel_factory: Any
    policy: ReviewPolicy = dfield(default_factory=ReviewPolicy)
    #: The run this review belongs to. Findings and ``review_round`` events are filed
    #: under it, so a run-filtered event listing shows the whole review.
    run_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.policy, dict):
            self.policy = ReviewPolicy(**self.policy)

    @staticmethod
    def ensure_separate(author_session: str, reviewer_session: str) -> None:
        if author_session == reviewer_session:
            raise SeparationOfDutyError(
                f"reviewer session {reviewer_session!r} must differ from author {author_session!r}"
            )

    #: Prefix that makes a session a reviewer session. Deterministic on purpose: a
    #: timestamped name can never collide, so separation of duty would be true by
    #: construction and untestable. Deriving it from the author's session means the
    #: model call is observably made under a different, checkable role.
    REVIEWER_PREFIX = "reviewer::"

    def _reviewer_session(self, author_session: str) -> str:
        candidate = f"{self.REVIEWER_PREFIX}{author_session}"
        self.ensure_separate(author_session, candidate)
        return candidate

    def review_plan(self, problem: "ProblemSpec", plan: "Plan", author_session: str) -> ReviewReport:
        reviewer = self._reviewer_session(author_session)
        from sherpa.ir import validate_plan

        concerns = concern_ledger(problem)
        sha = ledger_sha(concerns)
        started = time.time()
        findings: list[Finding] = []
        deterministic_findings: list[Finding] = []

        for err in validate_plan(plan):
            deterministic_findings.append(self._persist_finding(
                criterion=f"plan_validity:{err.code}", subject=f"plan:{plan.id}@{plan.version}",
                evidence_ref=f"{err.path}: {err.message}", blocking=True,
            ))
        if not problem.authority.allows(plan.authority):
            deterministic_findings.append(self._persist_finding(
                criterion="authority_containment", subject=f"plan:{plan.id}@{plan.version}",
                evidence_ref="plan authority exceeds problem-delegated grants", blocking=True,
            ))
        child_b = plan.budgets
        parent_b = problem.budgets
        over_budget = (
            child_b.max_nodes > parent_b.max_nodes
            or child_b.max_tokens > parent_b.max_tokens
            or child_b.max_depth > parent_b.max_depth
            or child_b.max_wall_seconds > parent_b.max_wall_seconds
        )
        if over_budget:
            deterministic_findings.append(self._persist_finding(
                criterion="budget_containment", subject=f"plan:{plan.id}@{plan.version}",
                evidence_ref=json.dumps({"child": child_b.model_dump(), "parent": parent_b.model_dump()}),
                blocking=True, rationale="plan budgets exceed parent budgets",
            ))

        findings.extend(deterministic_findings)

        subject = f"plan:{plan.id}@{plan.version}"
        model_findings, model_rounds, model_tokens, channel_error = self._model_rounds(
            system=(
                "You are an independent plan reviewer. Return strict JSON: "
                '{"findings":[{"criterion":str,"subject":str,"evidence":str|null,"blocking":bool}]}. '
                "A blocking finding REQUIRES concrete reproducible evidence. "
                + ledger_prompt(concerns, sha)
            ),
            payload=plan.model_dump_json(),
            subject=subject,
            reviewer=reviewer,
            started=started,
            allowed_criteria=ledger_criteria(concerns),
        )
        findings.extend(model_findings)
        if channel_error:
            findings.append(self._channel_finding(subject, channel_error))

        self._log_round(problem_id=problem.id, subject=subject,
                        round_no=model_rounds, n_findings=len(findings), tokens=model_tokens,
                        reviewer=reviewer)
        return self._verdict(findings, model_rounds, model_tokens, sha, channel_error)

    def review_output(
        self,
        problem: "ProblemSpec",
        artifacts: dict[str, str],
        deterministic_results: dict[str, bool],
        author_session: str,
    ) -> ReviewReport:
        reviewer = self._reviewer_session(author_session)
        concerns = concern_ledger(problem)
        sha = ledger_sha(concerns)
        findings: list[Finding] = []
        subject = "output:" + problem.id

        for check_id, passed in sorted(deterministic_results.items()):
            if not passed:
                findings.append(self._persist_finding(
                    criterion=f"acceptance:{check_id}", subject=subject,
                    evidence_ref=f"acceptance check {check_id} failed on real execution", blocking=True,
                ))
        missing = sorted({c.id.replace("concern_", "") for c in concerns if c.source == "contract"}
                         - set(deterministic_results))
        if missing:
            findings.append(self._persist_finding(
                criterion="no_acceptance_check_skipped", subject=subject,
                evidence_ref=f"unexecuted acceptance checks: {missing}", blocking=True,
            ))
        findings.extend(self._check_outputs_schema(problem, artifacts, subject))
        findings.extend(self._check_budgets(problem, subject))

        model_findings, model_rounds, model_tokens, channel_error = self._model_rounds(
            system=(
                'You are an independent output reviewer. Strict JSON: '
                '{"findings":[{"criterion":str,"subject":str,"evidence":str|null,"blocking":bool}]}. '
                "Blocking requires reproducible evidence. "
                + ledger_prompt(concerns, sha)
            ),
            payload=json.dumps({"artifacts": list(artifacts), "deterministic_results": deterministic_results}),
            subject=subject,
            reviewer=reviewer,
            started=time.time(),
            allowed_criteria=ledger_criteria(concerns),
        )
        findings.extend(model_findings)
        if channel_error:
            findings.append(self._channel_finding(subject, channel_error))

        self._log_round(problem_id=problem.id, subject=subject, round_no=model_rounds,
                        n_findings=len(findings), tokens=model_tokens, reviewer=reviewer)
        return self._verdict(findings, model_rounds, model_tokens, sha, channel_error)

    def _check_outputs_schema(self, problem: "ProblemSpec", artifacts: dict[str, str],
                              subject: str) -> list[Finding]:
        """Generic concern ``outputs_conform_to_schema``: real check, real evidence."""
        from sherpa.admission import check_io

        raw = artifacts.get("outputs_json")
        if raw is None:
            payload: Any = dict(artifacts)
        else:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                return [self._persist_finding(
                    criterion="outputs_conform_to_schema", subject=subject,
                    evidence_ref=f"outputs_json is not decodable JSON: {exc}", blocking=True,
                )]
        ok, errors = check_io(payload, problem.output_schema)
        if ok:
            return []
        return [self._persist_finding(
            criterion="outputs_conform_to_schema", subject=subject,
            evidence_ref=json.dumps(
                {"schema": problem.output_schema, "outputs": payload, "errors": errors},
                sort_keys=True,
            ),
            blocking=True,
            rationale="declared output schema not satisfied by the produced outputs",
        )]

    def _check_budgets(self, problem: "ProblemSpec", subject: str) -> list[Finding]:
        """Generic concern ``budgets_respected``: recorded usage vs delegated budgets.

        Needs the run this review belongs to; without a run id there is no recorded
        usage to check against.
        """
        if not self.run_id:
            return []
        usage = self.store.usage(self.run_id)
        limits = {
            "tokens": float(problem.budgets.max_tokens),
            "nodes": float(problem.budgets.max_nodes),
            "wall_seconds": float(problem.budgets.max_wall_seconds),
            "cost_usd": float(problem.budgets.max_cost_usd),
        }
        over = {
            field: {"used": float(usage.get(field, 0.0)), "budget": limit}
            for field, limit in limits.items()
            if float(usage.get(field, 0.0)) > limit
        }
        if not over:
            return []
        return [self._persist_finding(
            criterion="budgets_respected", subject=subject,
            evidence_ref=json.dumps({"run_id": self.run_id, "over_budget": over}, sort_keys=True),
            blocking=True,
            rationale="recorded usage exceeded the delegated budgets",
        )]

    def _model_rounds(
        self,
        *,
        system: str,
        payload: str,
        subject: str,
        reviewer: str,
        started: float,
        allowed_criteria: set[str],
    ) -> tuple[list[Finding], int, int, str]:
        """Run the bounded reviewer rounds; report a channel failure instead of hiding it.

        Returns ``(findings, rounds, tokens, channel_error)``. ``channel_error`` is
        non-empty when the reviewer needed a model round and the boundary refused
        (no channel configured, recordings exhausted, no live provider) — the caller
        must escalate rather than report a pass.
        """
        findings: list[Finding] = []
        rounds = 0
        tokens = 0
        for _round in range(self.policy.max_rounds):
            if time.time() - started > self.policy.max_seconds or tokens >= self.policy.max_tokens:
                break
            try:
                resp = self.channel_factory().complete(
                    [{"role": "system", "content": system}, {"role": "user", "content": payload}],
                    session=reviewer,
                )
            except CHANNEL_FAILURES as exc:
                return findings, rounds, tokens, f"{type(exc).__name__}: {exc}"
            rounds += 1
            tokens += resp.prompt_tokens + resp.completion_tokens
            adjudicated = self._adjudicate(resp.text, subject=subject,
                                           allowed_criteria=allowed_criteria)
            findings.extend(adjudicated)
            if not any(f.blocking for f in adjudicated):
                break
        return findings, rounds, tokens, ""

    def _channel_finding(self, subject: str, channel_error: str) -> Finding:
        return self._persist_finding(
            criterion="review_channel_unavailable",
            subject=subject,
            evidence_ref=channel_error,
            blocking=False,
            disposition="deferred",
            rationale="review incomplete: a reviewer model round was required and the channel refused",
        )

    def _adjudicate(self, text: str, subject: str, *, allowed_criteria: set[str]) -> list[Finding]:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
        out: list[Finding] = []
        for raw in data.get("findings", []):
            criterion = str(raw.get("criterion", "uncategorized"))[:200]
            evidence = raw.get("evidence")
            blocking_claim = bool(raw.get("blocking"))
            # Whitespace is not reproducible evidence: a blocking finding needs a
            # criterion PLUS evidence someone else can re-run (#492 §4).
            evidence_ref = str(evidence).strip() if evidence is not None else ""
            if blocking_claim and not evidence_ref:
                blocking_claim = False
            off_ledger = criterion not in allowed_criteria
            if off_ledger:
                # Scope drift: the ledger was frozen at review start precisely so a
                # later round cannot invent a new concern. Recorded, never blocking.
                blocking_claim = False
            unevidenced = bool(raw.get("blocking")) and not evidence_ref
            if off_ledger:
                disposition, rationale = "invalid", (
                    "downgraded: criterion is outside the frozen concern ledger for this review"
                )
            elif unevidenced:
                disposition, rationale = "invalid", "downgraded: no reproducible evidence provided"
            else:
                disposition, rationale = "open", ""
            out.append(self._persist_finding(
                criterion=criterion, subject=subject,
                evidence_ref=evidence_ref, blocking=blocking_claim,
                disposition=disposition, rationale=rationale,
            ))
        return out

    def _persist_finding(
        self,
        *,
        criterion: str,
        subject: str,
        evidence_ref: str,
        blocking: bool,
        disposition: str = "open",
        rationale: str = "",
    ) -> Finding:
        if disposition not in DISPOSITIONS:
            raise ValueError(f"invalid disposition {disposition!r}")
        fid = _finding_id(criterion, subject, evidence_ref)
        finding = Finding(
            id=fid, criterion=criterion, subject=subject, evidence_ref=evidence_ref,
            blocking=blocking, disposition=disposition, rationale=rationale,
        )
        self.store.add_finding(
            {
                "id": finding.id,
                "run_id": self.run_id,
                "subject": subject,
                "criterion": criterion,
                "evidence_ref": evidence_ref,
                "blocking": blocking,
                "disposition": finding.disposition,
                "rationale": finding.rationale,
            }
        )
        return finding

    def _log_round(self, *, problem_id: str, subject: str, round_no: int, n_findings: int,
                   tokens: int, reviewer: str) -> None:
        from sherpa.events import Event

        self.store.append(
            Event(
                kind="review_round",
                run_id=self.run_id,
                payload={
                    "problem_id": problem_id,
                    "subject": subject,
                    "round": round_no,
                    "n_findings": n_findings,
                    "tokens": tokens,
                    "reviewer_session": reviewer,
                },
            )
        )

    def disposition(self, finding_id: str, disposition: str, rationale: str = "") -> bool:
        return self.store.set_finding_disposition(finding_id, disposition, rationale)

    def _verdict(self, findings: list[Finding], rounds: int, tokens: int, sha: str,
                 channel_error: str = "") -> ReviewReport:
        blocking_open = [f for f in findings if f.blocking and f.disposition == "open"]
        risks = [
            f for f in findings
            if (not f.blocking) and f.disposition in ("open", "invalid", "deferred")
        ]
        if blocking_open:
            verdict = VERDICT_BLOCKED
        elif channel_error:
            # The review could not be completed. Reporting a pass here would turn
            # "this run unexpectedly needed a model" into "looks fine".
            verdict = VERDICT_INCOMPLETE
        else:
            verdict = VERDICT_PASS
        return ReviewReport(
            verdict=verdict,
            findings=findings,
            residual_risks=risks,
            rounds=rounds,
            tokens=tokens,
            ledger_sha=sha,
            channel_error=channel_error,
        )


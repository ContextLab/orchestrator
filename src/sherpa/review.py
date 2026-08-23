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

if TYPE_CHECKING:
    from sherpa.channel import ModelChannel
    from sherpa.ir import Plan, ProblemSpec
    from sherpa.store import BlobStore, Store

DISPOSITIONS = ("open", "fixed", "accepted_risk", "invalid", "deferred", "superseded")


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


@dataclass
class Reviewer:
    store: "Store"
    blob: "BlobStore"
    channel_factory: Any
    policy: ReviewPolicy = dfield(default_factory=ReviewPolicy)

    def __post_init__(self) -> None:
        if isinstance(self.policy, dict):
            self.policy = ReviewPolicy(**self.policy)

    @staticmethod
    def ensure_separate(author_session: str, reviewer_session: str) -> None:
        if author_session == reviewer_session:
            raise SeparationOfDutyError(
                f"reviewer session {reviewer_session!r} must differ from author {author_session!r}"
            )

    def _reviewer_session(self, author_session: str) -> str:
        candidate = f"rev_{int(time.time() * 1000) % 10_000_000}_{id(self) % 100_000}"
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
                evidence="plan budgets exceed parent budgets",
                evidence_ref=json.dumps({"child": child_b.model_dump(), "parent": parent_b.model_dump()}),
                blocking=True,
            ))

        findings.extend(deterministic_findings)

        model_rounds = 0
        model_tokens = 0
        for round_no in range(max(1, self.policy.max_rounds)):
            if time.time() - started > self.policy.max_seconds or model_tokens >= self.policy.max_tokens:
                break
            try:
                channel = self.channel_factory()
                resp = channel.complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are an independent plan reviewer. Return strict JSON: "
                                '{"findings":[{"criterion":str,"subject":str,"evidence":str|null,"blocking":bool}]}. '
                                "A blocking finding REQUIRES concrete reproducible evidence."
                            ),
                        },
                        {"role": "user", "content": plan.model_dump_json()},
                    ],
                    session="reviewer",
                )
            except Exception:  # noqa: BLE001 - no channel configured: deterministic-only review
                break
            model_rounds += 1
            model_tokens += resp.prompt_tokens + resp.completion_tokens
            adjudicated = self._adjudicate(resp.text, subject=f"plan:{plan.id}@{plan.version}")
            findings.extend(adjudicated)
            if not any(f.blocking for f in adjudicated):
                break

        self._log_round(problem_id=problem.id, subject=f"plan:{plan.id}@{plan.version}",
                        round_no=model_rounds, n_findings=len(findings), tokens=model_tokens,
                        reviewer=reviewer)
        return self._verdict(findings, model_rounds, model_tokens, sha)

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

        model_rounds = 0
        model_tokens = 0
        started = time.time()
        for round_no in range(max(1, self.policy.max_rounds)):
            if time.time() - started > self.policy.max_seconds:
                break
            try:
                channel = self.channel_factory()
                payload = json.dumps({"artifacts": list(artifacts), "deterministic_results": deterministic_results})
                resp = channel.complete(
                    [
                        {"role": "system", "content": (
                            'You are an independent output reviewer. Strict JSON: '
                            '{"findings":[{"criterion":str,"subject":str,"evidence":str|null,"blocking":bool}]}. '
                            "Blocking requires reproducible evidence."
                        )},
                        {"role": "user", "content": payload},
                    ],
                    session="reviewer",
                )
            except Exception:  # noqa: BLE001 - deterministic-only review without a channel
                break
            model_rounds += 1
            model_tokens += resp.prompt_tokens + resp.completion_tokens
            adjudicated = self._adjudicate(resp.text, subject=subject)
            findings.extend(adjudicated)
            if not any(f.blocking for f in adjudicated):
                break

        self._log_round(problem_id=problem.id, subject=subject, round_no=model_rounds,
                        n_findings=len(findings), tokens=model_tokens, reviewer=reviewer)
        return self._verdict(findings, model_rounds, model_tokens, sha)

    def _adjudicate(self, text: str, subject: str) -> list[Finding]:
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
            evidence_ref = str(evidence) if evidence else ""
            if blocking_claim and not evidence_ref:
                blocking_claim = False
            out.append(self._persist_finding(
                criterion=criterion, subject=subject,
                evidence_ref=evidence_ref, blocking=blocking_claim,
                unevidenced=(bool(raw.get("blocking")) and not evidence_ref),
            ))
        return out

    def _persist_finding(
        self,
        *,
        criterion: str,
        subject: str,
        evidence_ref: str,
        blocking: bool,
        evidence: str | None = None,
        unevidenced: bool = False,
    ) -> Finding:
        fid = _finding_id(criterion, subject, evidence_ref)
        finding = Finding(
            id=fid, criterion=criterion, subject=subject, evidence_ref=evidence_ref,
            blocking=blocking,
            disposition="invalid" if unevidenced else "open",
            rationale="downgraded: no reproducible evidence provided" if unevidenced else "",
        )
        self.store.add_finding(
            {
                "id": finding.id,
                "run_id": "",
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
                run_id=problem_id,
                payload={
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

    def _verdict(self, findings: list[Finding], rounds: int, tokens: int, sha: str) -> ReviewReport:
        blocking_open = [f for f in findings if f.blocking and f.disposition == "open"]
        risks = [
            f for f in findings
            if (not f.blocking) and f.disposition in ("open", "invalid")
        ]
        return ReviewReport(
            verdict="blocked_escalated" if blocking_open else "pass_with_risk",
            findings=findings,
            residual_risks=risks,
            rounds=rounds,
            tokens=tokens,
            ledger_sha=sha,
        )


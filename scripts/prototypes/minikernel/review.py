"""The review protocol: evidential gate + frozen concern ledger.

Three mechanisms, each answering a measured failure mode:

  M1  SEPARATION OF DUTY. A session may not review an artifact version it
      authored. Enforced on session identity, not on politeness.

  M2  FROZEN CRITERIA + CONCERN LEDGER. Acceptance criteria are frozen before
      authoring. Every finding must cite a frozen criterion; one that cites
      none is auto-filed `deferred`, never `open`. This is what stops scope
      drift from turning a 3.6-round loop into an 8.6-round loop with a
      non-terminating tail (round-1 simulation, red_team_gate.py).

  M3  EVIDENTIAL GATE. A finding may only BLOCK if it ships a reproducible
      artifact -- a failing check with an observed value. A prose worry is
      recorded as a `risk`: visible, attached to the artifact forever, but
      non-blocking. This is what makes the loop terminate on evidence rather
      than on the reviewer's imagination running out, and it is what makes
      reviewer detection rate measurable (you can count artifacts).

What the gate does NOT do is certify correctness. `P(truly clean | passed)` is
bounded by the reviewer's detection rate, and no gate policy raises it. The
kernel therefore records `residual_risk` on every pass rather than reporting
"clean".
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from .capabilities import SeparationOfDuty
from .store import Store, content_hash

_ids = itertools.count(1)


@dataclass(frozen=True)
class Criterion:
    id: str
    text: str
    check: Callable[[Any], tuple[bool, str]] | None = None


@dataclass
class Finding:
    id: str
    criterion_id: str | None
    severity: str                      # blocker | major | minor
    summary: str
    evidence: str | None               # None => prose worry => risk, non-blocking
    status: str = "open"               # open|fixed|accepted_risk|invalid|deferred|risk
    round: int = 0

    @property
    def blocking(self) -> bool:
        return (
            self.status == "open"
            and self.evidence is not None
            and self.criterion_id is not None
            and self.severity in ("blocker", "major")
        )


@dataclass
class ArtifactVersion:
    artifact_id: str
    version: int
    body: str
    author_session: str

    @property
    def ref(self) -> str:
        return f"{self.artifact_id}@v{self.version}"

    @property
    def hash(self) -> str:
        return content_hash(self.body)


@dataclass
class ReviewOutcome:
    passed: bool
    rounds: int
    findings: list[Finding]
    escalated: bool = False
    residual_risk: float = 0.0
    reviewer_tokens: int = 0

    def open_blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.blocking]

    def risks(self) -> list[Finding]:
        return [f for f in self.findings if f.status in ("risk", "deferred")]


class ReviewBoard:
    def __init__(self, store: Store, max_rounds: int = 6,
                 detection_rate_prior: float = 0.6):
        self.store = store
        self.max_rounds = max_rounds
        self.detection_rate_prior = detection_rate_prior
        self.ledger: dict[str, list[Finding]] = {}

    def freeze(self, artifact_id: str, criteria: list[Criterion]) -> None:
        self.store.append_event(
            "review", artifact_id, "criteria_frozen",
            {"criteria": [{"id": c.id, "text": c.text} for c in criteria]},
        )
        self._frozen = getattr(self, "_frozen", {})
        self._frozen[artifact_id] = {c.id: c for c in criteria}

    def frozen(self, artifact_id: str) -> dict[str, Criterion]:
        return getattr(self, "_frozen", {}).get(artifact_id, {})

    def file(self, artifact: ArtifactVersion, reviewer_session: str,
             finding: Finding) -> Finding:
        """Classify a submitted finding against the frozen criteria (M2, M3)."""
        frozen = self.frozen(artifact.artifact_id)
        if finding.criterion_id not in frozen:
            finding.status = "deferred"        # out of frozen scope: cannot block
            finding.criterion_id = None
        elif finding.evidence is None:
            finding.status = "risk"            # prose worry: visible, non-blocking
        self.ledger.setdefault(artifact.artifact_id, []).append(finding)
        self.store.append_event(
            "review", artifact.ref, "finding_filed",
            {"id": finding.id, "criterion": finding.criterion_id,
             "severity": finding.severity, "status": finding.status,
             "has_evidence": finding.evidence is not None,
             "summary": finding.summary[:200]},
            session_id=reviewer_session,
        )
        return finding

    def run(
        self,
        artifact: ArtifactVersion,
        criteria: list[Criterion],
        reviewer_session: str,
        revise: Callable[[ArtifactVersion, list[Finding]], ArtifactVersion],
        reviewer: Callable[[ArtifactVersion, list[Criterion], int], list[Finding]] | None = None,
    ) -> tuple[ArtifactVersion, ReviewOutcome]:
        """Author <-> reviewer loop, bounded, over immutable artifact versions."""
        if reviewer_session == artifact.author_session:
            raise SeparationOfDuty(
                f"session {reviewer_session!r} authored {artifact.ref}"
            )
        self.freeze(artifact.artifact_id, criteria)
        reviewer = reviewer or self._default_reviewer
        rounds = 0
        all_findings: list[Finding] = []
        tokens = 0
        current = artifact
        while rounds < self.max_rounds:
            rounds += 1
            submitted = reviewer(current, criteria, rounds)
            tokens += 1200 + 400 * len(submitted)
            classified = [self.file(current, reviewer_session, f) for f in submitted]
            all_findings += classified
            blockers = [f for f in classified if f.blocking]
            if not blockers:
                residual = self.detection_rate_prior
                outcome = ReviewOutcome(
                    True, rounds, all_findings, False,
                    residual_risk=round(1.0 - residual, 3), reviewer_tokens=tokens,
                )
                self.store.append_event(
                    "review", current.ref, "review_passed",
                    {"rounds": rounds, "risks": len(outcome.risks()),
                     "residual_risk": outcome.residual_risk},
                    session_id=reviewer_session,
                )
                return current, outcome
            current = revise(current, blockers)
            for f in blockers:
                f.status = "fixed"
            self.store.append_event(
                "review", current.ref, "artifact_revised",
                {"fixed": [f.id for f in blockers], "version": current.version},
                session_id=current.author_session,
            )
        outcome = ReviewOutcome(
            False, rounds, all_findings, escalated=True,
            residual_risk=1.0, reviewer_tokens=tokens,
        )
        self.store.append_event(
            "review", current.ref, "review_escalated",
            {"rounds": rounds, "open": len(outcome.open_blockers())},
            session_id=reviewer_session,
        )
        return current, outcome

    def _default_reviewer(
        self, artifact: ArtifactVersion, criteria: list[Criterion], round_: int
    ) -> list[Finding]:
        """Executable-criteria reviewer: run every check, report what fails.

        This is the evidential gate in its purest form -- the reviewer cannot
        express a concern it cannot demonstrate.
        """
        out: list[Finding] = []
        for c in criteria:
            if c.check is None:
                continue
            ok, observed = c.check(artifact.body)
            if not ok:
                out.append(
                    Finding(
                        id=f"F{next(_ids)}", criterion_id=c.id, severity="blocker",
                        summary=f"criterion {c.id} not satisfied",
                        evidence=observed, round=round_,
                    )
                )
        return out


# ------------------------------------------------------------ insight pool


class InsightPool:
    """Insights are notes with `kind='insight'`; the pool is the gate around them.

    #485 requires an independent red-team pass on both insertion and removal.
    Added here: contradiction detection at insertion (two accepted insights that
    disagree will otherwise both be retrieved and silently degrade every
    downstream agent) and a scope tag so a run-local insight cannot leak.
    """

    def __init__(self, store: Store, board: ReviewBoard):
        self.store = store
        self.board = board

    def propose(self, run_id: str, node_id: str, text: str, author_session: str,
                reviewer_session: str, scope: str = "run") -> tuple[bool, str]:
        if reviewer_session == author_session:
            raise SeparationOfDuty("insights are reviewed by a different session")
        contradiction = self._contradicts(text)
        if contradiction is not None:
            self.store.append_event(
                run_id, node_id, "insight_rejected",
                {"reason": "contradicts", "conflicts_with": contradiction},
                session_id=reviewer_session,
            )
            return False, f"contradicts accepted insight {contradiction}"
        seq = self.store.append_note(run_id, node_id, "insight", text,
                                     session_id=author_session, scope=scope)
        self.store.append_event(run_id, node_id, "insight_accepted",
                                {"seq": seq, "scope": scope},
                                session_id=reviewer_session)
        return True, str(seq)

    _NEGATIONS = (" not ", " never ", " no ", " cannot ", " isn't ", " doesn't ")
    _AUX = {"does", "did", "will", "would", "shall", "should", "must",
            "have", "has", "had", "been", "being", "were", "was"}

    @staticmethod
    def _stem(w: str) -> str:
        for suf in ("ing", "ies", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                return w[: -len(suf)]
        return w

    def _contradicts(self, text: str) -> str | None:
        """Cheap polarity check over shared content words.

        Deliberately crude: the point of the experiment is whether the pool
        NEEDS contradiction detection, not whether this detector is good.

        FIX (found by scenario S9): the first version compared raw surface
        forms, so "drifts" and "does not drift" shared only half their tokens
        and the contradiction sailed through. Two accepted insights that
        disagree are worse than no insight at all -- both get retrieved, and
        every downstream agent quietly averages them.
        """
        def core(s: str) -> set[str]:
            low = f" {s.lower()} "
            for n in self._NEGATIONS:
                low = low.replace(n, " ")
            return {self._stem(w) for w in low.split()
                    if len(w) > 3 and w not in self._AUX}

        def polarity(s: str) -> bool:
            return any(n in f" {s.lower()} " for n in self._NEGATIONS)

        mine, mypol = core(text), polarity(text)
        for row in self.store.view_insights():
            body = self.store.get_blob(row["body_hash"]) or ""
            theirs = core(body)
            if not theirs:
                continue
            overlap = len(mine & theirs) / max(1, len(mine | theirs))
            if overlap > 0.6 and polarity(body) != mypol:
                return f"note:{row['seq']}"
        return None

    def retract(self, seq: int, author_session: str, reviewer_session: str,
                reason: str) -> bool:
        if reviewer_session == author_session:
            raise SeparationOfDuty("retractions are reviewed by a different session")
        self.store._db.execute(
            "UPDATE notes SET status='retracted' WHERE seq=?", (seq,)
        )
        self.store._db.commit()
        self.store.append_event("registry", f"note:{seq}", "insight_retracted",
                                {"reason": reason}, session_id=reviewer_session)
        return True

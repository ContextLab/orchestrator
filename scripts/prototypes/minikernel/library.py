"""The solved-problem library -- the termination mechanism, not a cache.

#485 files "compound engineering" under component 3, as an efficiency story.
The branching analysis says it is actually the control system for component 1:
recursion terminates in expectation iff `m = b*f < 1`, and every solved subtree
that lands in the library converts a future ambiguous step into an atomic one,
which is the only mechanism in the design that drives `f` down. A system that
starts supercritical becomes subcritical by remembering.

Retrieval keys on TWO things, because semantic similarity alone produces the
"close enough, wrong shape" failure: a normalised statement similarity AND a
compatible typed I/O signature. Both must pass.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .ir import Plan
from .store import Store, content_hash

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "by",
    "from", "that", "this", "it", "is", "are", "be", "as", "at", "into", "then",
}


def tokens_of(statement: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", statement.lower())
        if len(w) > 2 and w not in _STOP
    }


def similarity(a: str, b: str) -> float:
    ta, tb = tokens_of(a), tokens_of(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def signature_compatible(want: str, have: str) -> bool:
    """`in1,in2->out`. `any` is a wildcard on either side of the arrow."""
    def parse(sig: str) -> tuple[list[str], str]:
        lhs, _, rhs = sig.partition("->")
        return [p.strip() for p in lhs.split(",") if p.strip()], rhs.strip() or "any"

    wi, wo = parse(want)
    hi, ho = parse(have)
    if wo != "any" and ho != "any" and wo != ho:
        return False
    if len(wi) != len(hi):
        return False
    return all(x == "any" or y == "any" or x == y for x, y in zip(wi, hi))


@dataclass
class Intractable:
    """A recorded NEGATIVE result.

    Found by scenario S7: a mission that ends at the depth cap teaches the
    system nothing, so re-running it costs exactly as much as the first time,
    forever. The library must remember what it could NOT solve, and under what
    allowance -- otherwise the only regime where learning matters is the one
    regime where learning cannot start.
    """

    key: str
    statement: str
    signature: str
    reason: str
    depth_allowance: int
    attempts: int = 1


@dataclass
class Solution:
    key: str
    statement: str
    signature: str
    plan_json: str
    evidence: dict[str, Any] = field(default_factory=dict)
    uses: int = 0
    wins: int = 0
    provenance: str = ""

    @property
    def reliability(self) -> float:
        """Laplace-smoothed success rate: an unproven entry is not a sure thing."""
        return (self.wins + 1) / (self.uses + 2)


class SolvedProblemLibrary:
    def __init__(self, store: Store, sim_threshold: float = 0.55,
                 min_reliability: float = 0.4):
        self.store = store
        self.sim_threshold = sim_threshold
        self.min_reliability = min_reliability
        self.entries: dict[str, Solution] = {}
        self.dead_ends: dict[str, Intractable] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def publish(self, statement: str, signature: str, plan: Plan,
                evidence: dict[str, Any], provenance: str = "") -> str | None:
        """Publish a solved problem -- if its shape is concrete enough to reuse.

        FIX (found by scenario S7): the two-key match is only as strong as the
        weaker key. A solution stored with signature `any->any` matches every
        later query, so text similarity silently becomes the ONLY key and the
        "close enough, wrong shape" failure comes straight back. An untyped
        solution is therefore not published at all.
        """
        from .ir import as_jsonable

        _, _, out = signature.partition("->")
        if out.strip() in ("", "any"):
            self.store.append_event(
                "library", content_hash(statement), "solution_not_published",
                {"reason": "wildcard output signature; not safely reusable",
                 "signature": signature, "statement": statement[:200]},
            )
            return None
        key = content_hash(f"{sorted(tokens_of(statement))}|{signature}")
        if key in self.entries:
            return key
        sol = Solution(key, statement, signature,
                       json.dumps(as_jsonable(plan), sort_keys=True),
                       evidence, provenance=provenance)
        self.entries[key] = sol
        self.store.append_event(
            "library", key, "solution_published",
            {"statement": statement[:200], "signature": signature,
             "evidence": evidence, "provenance": provenance},
        )
        return key

    def lookup(self, statement: str, signature: str) -> Solution | None:
        best, best_sim = None, 0.0
        for sol in self.entries.values():
            if not signature_compatible(signature, sol.signature):
                continue
            if sol.reliability < self.min_reliability:
                continue
            s = similarity(statement, sol.statement)
            if s >= self.sim_threshold and s > best_sim:
                best, best_sim = sol, s
        if best is not None:
            self.store.append_event(
                "library", best.key, "solution_hit",
                {"query": statement[:200], "similarity": round(best_sim, 3),
                 "reliability": round(best.reliability, 3)},
            )
        return best

    # ------------------------------------------------------ negative results

    @staticmethod
    def _neg_key(statement: str, signature: str) -> str:
        return content_hash(f"NEG|{sorted(tokens_of(statement))}|{signature}")

    def record_intractable(self, statement: str, signature: str, reason: str,
                           depth_allowance: int) -> str:
        key = self._neg_key(statement, signature)
        rec = self.dead_ends.get(key)
        if rec is None:
            rec = Intractable(key, statement, signature, reason, depth_allowance)
            self.dead_ends[key] = rec
        else:
            rec.attempts += 1
            rec.depth_allowance = max(rec.depth_allowance, depth_allowance)
        self.store.append_event(
            "library", key, "intractable_recorded",
            {"statement": statement[:200], "signature": signature,
             "reason": reason, "depth_allowance": depth_allowance,
             "attempts": rec.attempts},
        )
        return key

    def lookup_intractable(self, statement: str, signature: str,
                           depth_allowance: int) -> Intractable | None:
        """A dead end only counts if the earlier attempt had AT LEAST as much
        room as this one. More budget is a legitimate reason to try again."""
        for rec in self.dead_ends.values():
            if not signature_compatible(signature, rec.signature):
                continue
            if rec.depth_allowance < depth_allowance:
                continue
            if similarity(statement, rec.statement) >= self.sim_threshold:
                self.store.append_event("library", rec.key, "intractable_hit",
                                        {"query": statement[:200],
                                         "attempts": rec.attempts})
                return rec
        return None

    def record_use(self, key: str, outcome: str) -> None:
        """Record evidence about a reused plan.

        FIX (found by scenario S7): reliability must only move on evidence
        ABOUT THE PLAN. A run cut short by budget or by the depth cap says
        nothing about whether the cached plan was right, but counting it as a
        failure drops reliability below the retrieval floor after a single
        unlucky mission -- so budget pressure silently un-learns the library.
        """
        sol = self.entries.get(key)
        if sol is None:
            return
        if outcome in ("budget_exhausted", "escalated"):
            self.store.append_event("library", key, "solution_use_uncounted",
                                    {"outcome": outcome})
            return
        sol.uses += 1
        sol.wins += int(outcome == "completed")
        self.store.append_event("library", key, "solution_used",
                                {"outcome": outcome, "uses": sol.uses,
                                 "reliability": round(sol.reliability, 3)})

    def stats(self) -> dict[str, Any]:
        if not self.entries:
            return {"size": 0, "dead_ends": len(self.dead_ends),
                    "mean_reliability": 0.0, "total_uses": 0}
        return {
            "size": len(self.entries),
            "dead_ends": len(self.dead_ends),
            "mean_reliability": round(
                sum(s.reliability for s in self.entries.values()) / len(self.entries), 3
            ),
            "total_uses": sum(s.uses for s in self.entries.values()),
        }

"""Scenario C fixtures: an oversized corpus with seeded needles and distractors.

The corpus is far larger than the demo model channel's working context; every
needle is a unique verifiable fact embedded once. Retrieval must find needles
(FTS5), and every answer claim must resolve to immutable chunk spans.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

TOPICS = (
    "telescope calibration logs", "greenhouse temperature notes", "crew meal inventories",
    "orbital mechanics scratch work", "antenna maintenance reports", "dust storm advisories",
    "spectrometer readouts", "recreation schedules", "water recycling audits",
    "cargo manifests",
)

FILLER_SENTENCES = (
    "Readings were within nominal range for the fourth consecutive cycle.",
    "The committee agreed to revisit the schedule after the next supply drop.",
    "Two spare gaskets were logged into storage bay three without incident.",
    "Calibration drifted slightly under peak load but recovered overnight.",
    "The quarterly review highlighted steady progress on routine maintenance.",
    "A brief interruption in comms was traced to a misaligned relay.",
    "Morale remained high despite the extended dust season.",
    "Inventory reconciliation found no discrepancies this period.",
)


@dataclass(frozen=True)
class Corpus:
    docs: dict[str, str]
    needles: dict[str, tuple[str, str]]   # fact -> (doc_id, exact sentence)


def make_corpus(n_docs: int = 40, n_needles: int = 8, seed: int = 17) -> Corpus:
    rng = random.Random(seed)
    codes = [f"PERIDOT-{rng.randrange(10, 99)}" for _ in range(n_needles)]
    while len(set(codes)) != n_needles:
        codes = list(dict.fromkeys(codes + [f"PERIDOT-{rng.randrange(10, 99)}"]))
    codes = codes[:n_needles]

    doc_ids = [f"mission_log_{i:03d}" for i in range(n_docs)]
    rng.shuffle(doc_ids)
    needles: dict[str, tuple[str, str]] = {}
    for i, code in enumerate(codes):
        doc_id = doc_ids[(i * 7) % n_docs]
        sentence = f"During shift {i}, the duty officer confirmed the launch code was {code}."
        needles[code] = (doc_id, sentence)

    docs: dict[str, str] = {}
    for d_i, doc_id in enumerate(doc_ids):
        parts: list[str] = [f"# Mission log {d_i:03d}"]
        facts_here = [s for code, (did, s) in needles.items() if did == doc_id]
        body_len = 0
        s_i = 0
        while body_len < 2200:  # chars; total corpus >> any single-context demo window
            if facts_here and s_i % 37 == 18:
                parts.append(facts_here.pop())
            else:
                parts.append(FILLER_SENTENCES[rng.randrange(len(FILLER_SENTENCES))])
            body_len += 70
            s_i += 1
        while facts_here:
            parts.append(facts_here.pop())
        docs[doc_id] = "\n\n".join(parts) + "\n"
    return Corpus(docs=docs, needles=needles)


QUESTION = "List every launch code recorded across the mission logs, citing its document."

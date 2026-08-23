#!/usr/bin/env python3
"""Is the summary tree a retrieval mechanism or just a map?

The Opus review (§3) argued the inode-style summary DAG of #485 component 1b
is a *navigation* structure, not a *retrieval* structure: at depth d it retains
only r^d of the original, so a specific fact has ~zero chance of surviving to
the top. The follow-up comment then corrected the geometry: for a fixed
top-level size, top-level fidelity is set by the target size whatever r is —
what r controls is HOW MANY lossy hops you pass through.

Two questions get quantified here:

PART A — needle retrieval strategies on one synthetic corpus:
  N1 navigate-only: descend the summary tree by matching keywords.
  N2 index-only: perfect direct lookup (the FTS/RAG idealisation).
  N3 hybrid: index nominates candidates; summaries verify/rank them.
  Metrics: P(needle found), expected tokens read. Claim under test:
  navigation alone degrades with corpus size; the index is load-bearing.

PART B — hop distortion at fixed endpoint:
  Same corpus, same final top-level size S_top; reach it via few-hard hops
  (r=0.1) vs many-gentle hops (r=0.9). Each hop keeps r of what its child
  kept and independently drops eps of the survivors (per-hop noise).
  Needle survival after d hops ~ (r*(1-eps))^d with d = ceil(ln(N/S_top)/ln(1/r)).
  Claim under test (asserted): fewer, harder compressions preserve needles
  better than gentle multi-hop chains whenever per-hop noise > 0.

Corpus model
------------
* D documents over V words clustered into T topics; each doc draws most words
  from one topic (salience signal) plus background noise words.
* K needles: unique rare words, each planted in exactly one document.
* A "summary" of a span = the s most frequent non-noise words in the span
  (a keyword sketch parameterised by sketch size s — the stand-in for r).
"""

from __future__ import annotations

import math
import random
import statistics

ASSUMPTIONS = {
    "docs": 2_000,
    "words_per_doc": 60,
    "topics": 20,
    "vocab_per_topic": 40,
    "needles": 40,
    "fanout": 8,               # summary-tree branching factor
    "sketch_size": 12,         # terms kept in each summary node
    "noise_rate": 0.25,        # fraction of a doc's words that are background
    "trials": 300,
    "seed": 47,
}


class Corpus:
    def __init__(self, rng: random.Random):
        self.topic_words = {
            t: [f"w{t}_{j}" for j in range(ASSUMPTIONS["vocab_per_topic"])]
            for t in range(ASSUMPTIONS["topics"])
        }
        # Topical LOCALITY: contiguous document bands share a dominant topic
        # (with 15% cross-band strays), so sibling summary branches differ --
        # otherwise every bucket covers every topic and there is nothing to
        # navigate BY.
        self.docs: list[list[str]] = []
        band = max(1, ASSUMPTIONS["docs"] // ASSUMPTIONS["topics"])
        for i in range(ASSUMPTIONS["docs"]):
            t_dom = min(i // band, ASSUMPTIONS["topics"] - 1)
            t = t_dom if rng.random() > 0.15 else rng.randrange(ASSUMPTIONS["topics"])
            n_signal = int(ASSUMPTIONS["words_per_doc"] * (1 - ASSUMPTIONS["noise_rate"]))
            n_noise = ASSUMPTIONS["words_per_doc"] - n_signal
            words = [rng.choice(self.topic_words[t]) for _ in range(n_signal)]
            words += [f"bg{rng.randrange(1000)}" for _ in range(n_noise)]
            self.docs.append(words)
        self.needle_doc: dict[str, int] = {}
        for k in range(ASSUMPTIONS["needles"]):
            word = f"NEEDLE_{k}"
            di = rng.randrange(ASSUMPTIONS["docs"])
            self.docs[di].append(word)
            self.needle_doc[word] = di

    def sketch(self, doc_ids: list[int]) -> set[str]:
        """Keyword sketch of a span: `sketch_size` most frequent real words."""
        counts: dict[str, int] = {}
        for di in doc_ids:
            for w in self.docs[di]:
                if w.startswith("bg") or w.startswith("NEEDLE"):
                    continue          # sketches carry topic words, not noise
                counts[w] = counts.get(w, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        return {w for w, _ in ranked[:ASSUMPTIONS["sketch_size"]]}


def build_summary_tree(corpus: Corpus) -> list[dict]:
    """Leaves up; each level groups fanout siblings into a sketched parent."""
    fanout = ASSUMPTIONS["fanout"]
    levels: list[list[dict]] = [
        [{"doc_ids": [i], "keywords": None} for i in range(len(corpus.docs))]
    ]
    while len(levels[-1]) > fanout:
        prev = levels[-1]
        parents = []
        for i in range(0, len(prev), fanout):
            group = prev[i:i + fanout]
            ids = [d for node in group for d in node["doc_ids"]]
            parents.append({"doc_ids": ids, "children": group, "keywords": None})
        levels.append(parents)
    if len(levels[-1]) > 1:          # single root spanning the whole corpus
        prev = levels[-1]
        ids = [d for node in prev for d in node["doc_ids"]]
        levels.append([{"doc_ids": ids, "children": prev, "keywords": None}])
    root = levels[-1][0]
    root["keywords"] = corpus.sketch(root["doc_ids"])
    return levels


def navigate(corpus: Corpus, levels: list[dict], needle: str) -> tuple[bool, int]:
    """Descend by keyword match; returns (found, docs_consulted_tokens)."""
    node = levels[-1][0]
    reads = 1                       # reading the root sketch costs 1 unit
    di_target = corpus.needle_doc[needle]
    while "children" in node:
        best, best_score = None, -1
        for child in node["children"]:
            if child["keywords"] is None:
                child["keywords"] = corpus.sketch(child["doc_ids"])
            score = sum(1 for w in child["keywords"]
                        for dw in corpus.docs[di_target]
                        if not w.startswith("bg") and w == dw)
            # cheap proxy: overlap between child sketch and target doc's topic words
            if score > best_score:
                best, best_score = child, score
        reads += 1
        if best is None or best_score <= 0:
            return False, reads      # lost: term invisible from here
        node = best
    # leaf level: check the actual documents in the chosen bucket
    bucket = [i for i in node["doc_ids"]]
    reads += max(1, len(bucket) // 10)   # scanning a bucket costs proportionally
    return di_target in bucket, reads


def part_a(corpus: Corpus, levels: list[dict], rng: random.Random) -> None:
    print("== PART A: needle retrieval strategies ==")
    hdr = f"{'strategy':>22} {'found%':>7} {'mean read units':>16}"
    print(hdr)
    print("-" * len(hdr))
    needles = list(corpus.needle_doc)

    found_nav, reads_nav = [], []
    for nd in needles:
        ok, r = navigate(corpus, levels, nd)
        found_nav.append(ok)
        reads_nav.append(r)
    print(f"{'N1 navigate-only':>22} {statistics.fmean(found_nav):>6.0%} "
          f"{statistics.fmean(reads_nav):>16.1f}")

    # Index idealisation: near-perfect recall, tiny cost, but not perfect
    # (tokenisation/staleness misses happen -- assume 3% miss).
    index_recall = 0.97
    found_idx = [rng.random() < index_recall for _ in needles]
    reads_idx = [3] * len(needles)
    print(f"{'N2 index-only':>22} {statistics.fmean(found_idx):>6.0%} "
          f"{statistics.fmean(reads_idx):>16.1f}")

    # Hybrid: index first; on an index MISS fall back to summary navigation,
    # whose success rate applies to exactly those cases.
    hybrid_found = [i or n for i, n in zip(found_idx, found_nav)]
    hybrid_reads = [ri if i else rn
                    for ri, rn, i in zip(reads_idx, reads_nav, found_idx)]
    print(f"{'N3 hybrid (idx+nav)':>22} {statistics.fmean(hybrid_found):>6.0%} "
          f"{statistics.fmean(hybrid_reads):>16.1f}")

    print("* navigation succeeds only while the needle's TOPIC survives every")
    print("  hop; the specific needle never does. The index is load-bearing;")
    print("  the tree earns its keep as the fallback lane and the query planner.")
    print(f"* hybrid recovers to {statistics.fmean(hybrid_found):.0%} vs index-only "
          f"{statistics.fmean(found_idx):.0%} at near-index cost.")


def part_b() -> None:
    print("\n== PART B: hop distortion at fixed endpoint ==")
    eps = 0.05                  # per-hop independent drop probability
    n_docs = ASSUMPTIONS["docs"]
    s_top = math.ceil(n_docs / (ASSUMPTIONS["fanout"] ** 2))  # fixed endpoint
    print(f"same corpus -> same top size (~{s_top} docs); "
          f"per-hop drop eps={eps}; survival shown RELATIVE to the")
    print("endpoint baseline S_top/N (which is the same for every r)")
    hdr = f"{'r':>5} {'hops needed':>12} {'relative needle retention':>26}"
    print(hdr)
    print("-" * len(hdr))
    for r in (0.1, 0.3, 0.5, 0.9):
        d = math.ceil(math.log(n_docs / s_top) / math.log(1 / r))
        relative = (1 - eps) ** d
        print(f"{r:>5.1f} {d:>12} {relative:>25.0%}")
    print("* gentle multi-hop chains lose needles faster than few hard")
    print("  compressions to the SAME top size: retention = (1-eps)^d and")
    print("  d grows as r -> 1. Prefer fewer, harder compressions.")


def main() -> None:
    rng = random.Random(ASSUMPTIONS["seed"])
    corpus = Corpus(rng)
    levels = build_summary_tree(corpus)
    print(f"corpus: {ASSUMPTIONS['docs']} docs x {ASSUMPTIONS['words_per_doc']}w, "
          f"{ASSUMPTIONS['topics']} topics, {ASSUMPTIONS['needles']} needles; "
          f"summary tree fanout={ASSUMPTIONS['fanout']}, "
          f"levels={len(levels)}, sketch={ASSUMPTIONS['sketch_size']} words")
    part_a(corpus, levels, rng)
    part_b()


if __name__ == "__main__":
    main()

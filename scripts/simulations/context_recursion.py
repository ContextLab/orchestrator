#!/usr/bin/env python3
"""Does the "context recursion" scheme in #485 converge, and what survives it?

#485 proposes an inode-like scheme: content that exceeds 25% of a model's
context is split, each piece summarised, the summaries stitched back together,
and the whole thing repeated on the summaries until the top fits. The claim is
that this lets content of *any* length fit into any model's context.

**What this answers**

1. Does it terminate, and at what depth and cost?
2. How much of the original survives at the top?
3. What is left of the context window once #485's four allocations are paid?

**Finding (report section 3).** The geometry is fine -- depth stays at or below
5 even for a 100M-token corpus, and total cost is about 1.5x the corpus. It
converges iff ``r < 1``, which any real summariser satisfies. The problem is
fidelity: at depth 4 with r=0.2 the top-level view retains r**4 = 0.0016 of the
original, so a *specific fact* is essentially invisible from the root. The
summary tree is a navigation structure, not a retrieval structure -- which
makes the search index load-bearing rather than a convenience.

A second finding, surfaced by hardening this script against its tests: the
fidelity at the top is ``payload / n_tokens`` **whatever r is** -- it is forced
by the target size, not chosen. What ``r`` actually controls is how many lossy
hops you pass through to reach that size: r=0.1 gets there in 2 hops, r=0.9
needs 29. So for a fixed final size, aggressive single-hop summarisation
accumulates less distortion than gentle multi-hop summarisation. Prefer fewer,
harder compressions.

**Assumptions** (see ASSUMPTIONS; replace with measurements, then re-run).
The structural conclusions -- convergence iff r<1, depth logarithmic in N,
fidelity r**depth -- do not depend on the values. The specific numbers do.
"""

from __future__ import annotations

import math

# --- assumptions -----------------------------------------------------------
# Every value here is a guess, not a measurement. `r` in particular should be
# measured against a real summariser before any of the numbers are quoted.
ASSUMPTIONS = {
    "r": 0.2,  # summary tokens / input tokens, per level
    "frac": 0.25,  # share of context a payload may occupy (#485's rule)
    "instr": 800,  # prompt overhead per summariser call, tokens
    "out_reserve_frac": 0.5,  # share of payload budget reserved for the output
    "depth_cap": 12,  # backstop so a non-converging config still returns
}

# #485's four context allocations, as literally specified.
BUDGET = [
    ("scratchpad, direct read (tail)", 0.05),
    ("scratchpad, recursive summaries", 0.10),
    ("insights RAG seed", 0.10),
    ("one recursed document/problem payload", 0.25),
]


def summary_tree(
    n_tokens: int,
    context: int,
    frac: float = ASSUMPTIONS["frac"],
    r: float = ASSUMPTIONS["r"],
    instr: int = ASSUMPTIONS["instr"],
    out_reserve_frac: float = ASSUMPTIONS["out_reserve_frac"],
    depth_cap: int = ASSUMPTIONS["depth_cap"],
) -> dict:
    """Build the summary tree for ``n_tokens`` of content under ``context``.

    The usable chunk is smaller than the nominal ``frac * context``: the
    summariser has to hold the chunk, its instructions, and its own output in
    one window. Ignoring that overstates capacity by 5-20%.

    Returns depth, per-level shape, summariser calls, total tokens moved, the
    size of the top-level view, and ``r ** depth`` (the share of the original
    still represented up there).
    """
    payload = frac * context
    chunk = payload - instr - out_reserve_frac * payload * r
    if chunk <= 0:
        raise ValueError(
            f"overheads ({instr} + reserve) exceed the payload budget ({payload:.0f})"
        )

    # Two different claims, easily conflated: the recursion terminates
    # mathematically iff r < 1, but the depth that takes can be absurd as r
    # approaches 1. Report both.
    if r < 1 and n_tokens > payload:
        depth_required = math.ceil(math.log(payload / n_tokens) / math.log(r))
    else:
        depth_required = 0 if n_tokens <= payload else math.inf

    levels: list[tuple[int, float, int, float]] = []
    current, depth, calls, tokens = float(n_tokens), 0, 0, 0.0
    hit_cap = False
    while current > payload:
        if depth >= depth_cap:
            hit_cap = True
            break
        n_chunks = math.ceil(current / chunk)
        calls += n_chunks
        tokens += current + n_chunks * instr + current * r
        nxt = current * r
        levels.append((depth + 1, current, n_chunks, nxt))
        current, depth = nxt, depth + 1

    return {
        "depth": depth,
        "levels": levels,
        "calls": calls,
        "tokens": tokens,
        "top": current,
        "fidelity": r**depth,
        "chunk": chunk,
        "payload": payload,
        "converges": r < 1,
        "depth_required": depth_required,
        "hit_cap": hit_cap,
    }


def budget_subtotal() -> float:
    """Fraction of context #485 spends before any work happens."""
    return sum(share for _, share in BUDGET)


def main() -> None:
    print("== context-recursion tree geometry ==")
    print("model context C, doc size N -> depth / summariser calls / tokens burned")
    hdr = (
        f"{'C':>9} {'N':>12} {'r':>5} {'depth':>6} {'calls':>8} "
        f"{'tokens_in+out':>14} {'top_tokens':>11} {'r^depth':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for context in (128_000, 1_000_000):
        for n_tokens in (250_000, 5_000_000, 100_000_000):
            for r in (0.1, 0.2, 0.4):
                t = summary_tree(n_tokens, context, r=r)
                print(
                    f"{context:>9,} {n_tokens:>12,} {r:>5} {t['depth']:>6} "
                    f"{t['calls']:>8,} {int(t['tokens']):>14,} "
                    f"{int(t['top']):>11,} {t['fidelity']:>9.2e}"
                )

    print()
    print("== termination condition (N=5M, C=1M) ==")
    print("  converges iff r < 1 -- but the depth that takes explodes as r -> 1,")
    print("  and fidelity r**depth is what is left of the original at the top.")
    print()
    hdr_t = f"  {'r':<5} {'converges':>10} {'depth needed':>13} {'fidelity':>10}  note"
    print(hdr_t)
    print("  " + "-" * (len(hdr_t) - 2))
    for r in (0.1, 0.5, 0.9, 0.99, 1.0, 1.1):
        t = summary_tree(5_000_000, 1_000_000, r=r)
        needed = t["depth_required"]
        if not t["converges"]:
            shown, fidelity, note = "never", "-", "summaries do not shrink"
        else:
            shown = f"{needed:.0f}"
            fidelity = f"{r ** needed:.2e}"
            note = (
                f"exceeds the depth cap of {ASSUMPTIONS['depth_cap']}"
                if needed > ASSUMPTIONS["depth_cap"]
                else ""
            )
        print(f"  {r:<5} {str(t['converges']):>10} {shown:>13} {fidelity:>10}  {note}")

    print()
    print("== usable chunk after overheads (C=1M, frac=0.25) ==")
    for r in (0.1, 0.2, 0.4):
        t = summary_tree(1, 1_000_000, r=r)
        print(
            f"  r={r}: nominal payload {int(t['payload']):,} -> usable chunk "
            f"{int(t['chunk']):,} ({100 * t['chunk'] / t['payload']:.0f}% of nominal)"
        )

    print()
    print("== per-agent context budget, as literally specified in #485 ==")
    for label, share in BUDGET:
        print(f"  {share * 100:>5.0f}%  {label}")
    subtotal = budget_subtotal()
    print(f"  {'-' * 5}")
    print(f"  {subtotal * 100:>5.0f}%  SUBTOTAL (before system prompt, tool schemas,")
    print("         pipeline spec, child outputs, and the agent's own reasoning)")
    print(f"  => {100 - subtotal * 100:.0f}% of context left for the actual work.")
    for context in (128_000, 200_000, 1_000_000):
        print(
            f"     C={context:>9,} -> {int(context * (1 - subtotal)):>9,} "
            "tokens of working room"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does #485's recursive problem breakdown terminate, and what does it cost?

Component 1 of #485 breaks a task into ~10 steps; each step is either atomic
(solve it now) or ambiguous (recurse, it becomes its own pipeline). That is a
Galton-Watson branching process: a node emits ``b`` steps, each ambiguous with
probability ``f``, so mean offspring is ``m = b * f``.

**Finding (report sections 1 and 5).** The process is finite in expectation
**iff m < 1**. At #485's stated target of about 10 steps per pipeline, fewer
than 1 step in 10 may be ambiguous -- at every level -- or the tree does not
close. Nothing in #485 enforces that.

Three consequences, in the report:

1. The atomic/ambiguous classifier is the load-bearing component of the whole
   design, and its false-"ambiguous" rate is the number to measure first.
2. The solved-problem library is the *termination mechanism*, not an
   efficiency feature: every solved subtree that lands in it converts future
   ambiguous steps into atomic ones, driving ``f`` down until the process goes
   subcritical.
3. A depth cap alone is not enough. It truncates silently and returns a
   plausible wrong answer. Budget credits that propagate down the tree, with
   explicit escalation on exhaustion, fail loudly instead.

The cost model shows where the tokens go: 34% to re-reading shared state before
any work happens, 57% to review, 8% to the work.

**Assumptions** in ASSUMPTIONS. ``m = b * f < 1`` is structural and holds for
any values; the leaf counts and dollar figures do not.
"""

from __future__ import annotations

import random
import statistics

# --- assumptions -----------------------------------------------------------
ASSUMPTIONS = {
    "depth_cap": 8,  # #485 has none; this keeps supercritical runs finite
    "node_cap": 2_000_000,  # "runaway" threshold
    "trials": 400,
    "seed": 7,
    "context_tokens": 200_000,
    "red_team_rounds": 2.5,  # see red_team_gate.py for where this comes from
    "red_team_tokens": 34_000,  # fresh session re-reads context each round
    "usd_per_mtok": 3.0,
    "steps_per_hour": 450.0,  # the locked ceiling from scratchpad_contention.py
}

# Per-node token cost, using #485's own context percentages on a 200k model.
NODE_COSTS = [
    ("read scratchpad tail + summaries (15%)", 30_000),
    ("insight RAG seed (10%)", 20_000),
    ("plan / decompose or execute", 12_000),
    ("write scratchpad note", 1_000),
]
SHARED_STATE_COSTS = 2  # the first two rows are shared-state re-reading


def branch(
    b: int,
    f: float,
    rng: random.Random,
    depth_cap: int = ASSUMPTIONS["depth_cap"],
    node_cap: int = ASSUMPTIONS["node_cap"],
) -> tuple[int, int, int] | None:
    """One realisation of the decomposition tree.

    Starts from a single ambiguous root. Each ambiguous node emits ``b``
    children, each of which is itself ambiguous with probability ``f`` unless
    ``depth_cap`` forces it atomic. Returns (leaves, internal nodes, max depth),
    or None if the tree blew past ``node_cap`` before the cap bit.
    """
    frontier = [0]
    leaves = internal = max_depth = 0
    while frontier:
        depth = frontier.pop()
        internal += 1
        max_depth = max(max_depth, depth)
        if internal + leaves > node_cap:
            return None
        for _ in range(b):
            if depth + 1 < depth_cap and rng.random() < f:
                frontier.append(depth + 1)
            else:
                leaves += 1
    return leaves, internal, max_depth


def regime(b: int, f: float) -> str:
    """Subcritical / critical / supercritical, by mean offspring m = b*f."""
    m = b * f
    if m < 1:
        return "subcritical"
    return "critical" if m == 1 else "SUPERCRITICAL"


def leaf_distribution(b: int, f: float, trials: int = ASSUMPTIONS["trials"],
                      seed: int = ASSUMPTIONS["seed"]) -> dict:
    """Median and p95 leaf count over ``trials`` independent trees."""
    rng = random.Random(seed)
    results = [branch(b, f, rng) for _ in range(trials)]
    finite = [r[0] for r in results if r is not None]
    runaway = sum(1 for r in results if r is None)
    return {
        "m": b * f,
        "regime": regime(b, f),
        "median": statistics.median(finite) if finite else float("nan"),
        "p95": sorted(finite)[int(0.95 * len(finite))] if finite else float("nan"),
        "runaway_pct": 100.0 * runaway / trials,
    }


def tokens_per_node() -> float:
    """Total tokens one node spends, review included."""
    return (
        sum(cost for _, cost in NODE_COSTS)
        + ASSUMPTIONS["red_team_rounds"] * ASSUMPTIONS["red_team_tokens"]
    )


def shared_state_share() -> float:
    """Fraction of a node's tokens spent re-reading shared state before working."""
    return sum(c for _, c in NODE_COSTS[:SHARED_STATE_COSTS]) / tokens_per_node()


def main() -> None:
    print("== does recursive breakdown terminate? ==")
    print("b = steps per pipeline, f = P(a step is 'ambiguous' -> recurse)")
    print("Branching process: mean offspring m = b*f. Finite in expectation iff m < 1.")
    print()
    hdr = (
        f"{'b':>4} {'f':>6} {'m=b*f':>7} {'regime':>13} "
        f"{'median leaves':>14} {'p95 leaves':>11} {'runaway%':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for b in (5, 10):
        for f in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
            d = leaf_distribution(b, f)
            print(
                f"{b:>4} {f:>6} {d['m']:>7.2f} {d['regime']:>13} "
                f"{d['median']:>14,.0f} {d['p95']:>11,.0f} {d['runaway_pct']:>8.0f}%"
            )
    print()
    print(
        f"  (depth hard-capped at {ASSUMPTIONS['depth_cap']}; 'runaway' = more than "
        f"{ASSUMPTIONS['node_cap']:,} nodes before the cap bit)"
    )
    print("  => with b=10 you need f < 0.10: more than 90% of every pipeline's steps")
    print("     must classify atomic, at EVERY level, or the tree does not close.")

    per_node = tokens_per_node()
    print()
    print("== token cost of one node, as #485 specifies it ==")
    for label, cost in NODE_COSTS:
        print(f"  {cost:>8,}  {label}")
    print(
        f"  {ASSUMPTIONS['red_team_rounds'] * ASSUMPTIONS['red_team_tokens']:>8,.0f}  "
        f"red-team ({ASSUMPTIONS['red_team_rounds']} rounds x "
        f"{ASSUMPTIONS['red_team_tokens']:,} in a fresh session)"
    )
    print(f"  {per_node:>8,.0f}  TOTAL per node")
    print(
        f"  of which {shared_state_share():.0%} is shared-state re-reading, "
        "before any work happens"
    )

    print()
    hdr2 = (
        f"{'nodes':>8} {'tokens':>14} {'$ @ $3/Mtok':>13} "
        f"{'wall-clock @ 450 steps/hr':>27}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for n in (10, 100, 1_000, 10_000):
        total = n * per_node
        usd = total * ASSUMPTIONS["usd_per_mtok"] / 1e6
        hours = n / ASSUMPTIONS["steps_per_hour"]
        print(f"{n:>8,} {int(total):>14,} {usd:>12,.0f} {hours:>25,.1f}h")

    worst = leaf_distribution(10, 0.30)
    cost = worst["median"] * per_node * ASSUMPTIONS["usd_per_mtok"] / 1e6
    print()
    print(
        f"  cross-reference: b=10, f=0.30 has a median of {worst['median']:,.0f} "
        f"leaves -> about ${cost:,.0f} and "
        f"{worst['median'] / ASSUMPTIONS['steps_per_hour']:,.0f}h for one question."
    )


if __name__ == "__main__":
    main()

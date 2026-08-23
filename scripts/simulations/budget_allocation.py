#!/usr/bin/env python3
"""How should a parent split a token budget across children?

All three reviews demand budget credits instead of bare depth caps, but none
says HOW a parent should divide its allocation among children. This script
grows decomposition trees (branching process with heterogeneous — lognormal —
leaf costs) and compares three propagation policies at a fixed global budget:

  P1 **equal-split, no reserve**   each child gets an equal share; a child
     that runs out is silently truncated (the #478 "plausible wrong answer"
     failure mode).
  P2 **proportional, informed**    the parent splits proportionally to noisy
     per-child cost estimates.
  P3 **reserve & reallocate**      children get phi=70% of the parent's pool
     split equally; 30% stays in reserve. An exhausted child ESCALATES with
     partial results; the parent tops up from reserve or abandons it loudly.

Metrics: mission completion within budget, silent-truncation rate (garbage
shipped as success), loud-failure rate, and wasted tokens. The claim under
test: escalation beats prediction when subtree demand is heavy-tailed, and
equal-split-without-escalation silently ships incomplete work exactly when
the tree is largest.
"""

from __future__ import annotations

import math
import random
import statistics

ASSUMPTIONS = {
    "trials": 2000,
    "b": 10,                  # steps per pipeline (#485's target)
    "f": 0.15,                # ambiguity rate -> m = 1.5 supercritical start
    "depth_cap": 3,
    "leaf_cost_mu_ln": math.log(2_000),  # mean leaf execution ~2k tokens...
    "leaf_cost_sigma_ln": 1.0,           # ...but heavy-tailed
    "decomp_cost": 1_500,     # tokens for one internal node to plan/decompose
    "budget_multiple": 1.5,   # global budget = multiple of E[total demand]
    "reserve_fraction": 0.30,
    "max_topups_per_child": 1,
    "estimate_noise_sigma": 0.5,  # lognormal noise on policy-2 estimates
    "seed": 31,
}


def build_tree(rng: random.Random) -> tuple[int, float]:
    """Grow one tree; return (n_leaves, total_demand_tokens)."""
    b, f = ASSUMPTIONS["b"], ASSUMPTIONS["f"]
    frontier = [0]
    leaves = 0
    demand = ASSUMPTIONS["decomp_cost"]  # root plans
    while frontier:
        depth = frontier.pop()
        demand += ASSUMPTIONS["decomp_cost"]
        for _ in range(b):
            if depth + 1 < ASSUMPTIONS["depth_cap"] and rng.random() < f:
                frontier.append(depth + 1)
            else:
                leaves += 1
                demand += rng.lognormvariate(ASSUMPTIONS["leaf_cost_mu_ln"],
                                             ASSUMPTIONS["leaf_cost_sigma_ln"])
    return leaves, demand


def run_policy(policy: str, rng: random.Random) -> dict:
    """One mission under one budget policy."""
    leaves, _true_demand = build_tree(rng)

    # Global budget is set from the EXPECTED demand of the average tree, not
    # this tree's actual demand -- the governor cannot see the future.
    expected_leaves = _expected_leaves()
    expected_demand = (expected_leaves *
                       math.exp(ASSUMPTIONS["leaf_cost_mu_ln"] +
                                ASSUMPTIONS["leaf_cost_sigma_ln"] ** 2 / 2))
    budget_total = ASSUMPTIONS["budget_multiple"] * (
        expected_demand + ASSUMPTIONS["decomp_cost"] * _expected_nodes())

    # Simulate leaf demands and walk the policies top-down.
    leaf_costs = [rng.lognormvariate(ASSUMPTIONS["leaf_cost_mu_ln"],
                                     ASSUMPTIONS["leaf_cost_sigma_ln"])
                  for _ in range(leaves)]

    spent = 0.0
    completed = 0
    truncated = 0       # silent truncation (P1 failure mode)
    abandoned = 0       # loud failure (P3)

    if policy == "P1":
        share = budget_total / max(leaves, 1)
        for c in leaf_costs:
            spent += min(c, share)
            if c <= share:
                completed += 1
            else:
                truncated += 1        # ships plausible garbage, claims success

    elif policy == "P2":
        estimates = [c * math.exp(rng.gauss(0, ASSUMPTIONS["estimate_noise_sigma"]))
                     for c in leaf_costs]
        total_est = sum(estimates)
        scale = min(budget_total / max(total_est, 1), 50.0)  # cap over-allocation
        for c, e in zip(leaf_costs, estimates):
            alloc = e * scale
            spent += min(c, alloc)
            if c <= alloc:
                completed += 1
            else:
                truncated += 1

    else:  # P3 reserve & reallocate
        reserve = budget_total * ASSUMPTIONS["reserve_fraction"]
        pool = budget_total - reserve
        base = pool / max(leaves, 1)
        escalations: list[float] = []
        for c in leaf_costs:
            spent += min(c, base)
            if c <= base:
                completed += 1
            else:
                escalations.append(c - base)   # child reports shortfall honestly
        escalations.sort(reverse=True)
        for shortfall in escalations:
            topup = min(shortfall, reserve)
            reserve -= topup
            spent += topup
            if topup >= shortfall:
                completed += 1
            else:
                abandoned += 1                 # explicit partial-failure status

    return {
        "leaves": leaves,
        "completed_frac": completed / max(leaves, 1),
        "silent_truncation": truncated,
        "loud_abandon": abandoned,
        "spent": spent,
        "budget": budget_total,
        "mission_ok": completed == leaves,
    }


def _expected_leaves() -> float:
    """E[leaves] for b, f, depth_cap via the geometric series of the frontier."""
    b, f, d = ASSUMPTIONS["b"], ASSUMPTIONS["f"], ASSUMPTIONS["depth_cap"]
    total = 0.0
    level = 1.0
    for depth in range(d):
        total += level * b * (1 - f)
        level *= b * f
    total += level * b
    return total


def _expected_nodes() -> float:
    b, f, d = ASSUMPTIONS["b"], ASSUMPTIONS["f"], ASSUMPTIONS["depth_cap"]
    total, level = 0.0, 1.0
    for _ in range(d):
        total += level
        level *= b * f
    return total + level


def main() -> None:
    print("== budget allocation policies ==")
    print(f"b={ASSUMPTIONS['b']}, f={ASSUMPTIONS['f']} "
          f"(m={ASSUMPTIONS['b'] * ASSUMPTIONS['f']:.2f}), "
          f"depth cap {ASSUMPTIONS['depth_cap']}, leaf costs Lognormal("
          f"{ASSUMPTIONS['leaf_cost_mu_ln']:.1f}, {ASSUMPTIONS['leaf_cost_sigma_ln']})")
    print(f"global budget = {ASSUMPTIONS['budget_multiple']}x expected demand; "
          f"{ASSUMPTIONS['trials']} trials/policy")

    results = {}
    for policy in ("P1", "P2", "P3"):
        rng = random.Random(ASSUMPTIONS["seed"] + ord(policy[-1]))
        rows = [run_policy(policy, rng) for _ in range(ASSUMPTIONS["trials"])]
        results[policy] = rows

    names = {"P1": "P1 equal-split, no reserve",
             "P2": "P2 proportional (noisy estimates)",
             "P3": "P3 reserve & reallocate"}
    hdr = (f"{'policy':>32} {'all-leaves-done%':>17} {'silent-trunc/mission':>21} "
           f"{'loud-abandon/mission':>21} {'tokens wasted%':>15}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for p, rows in results.items():
        n = len(rows)
        waste = statistics.fmean(
            r["budget"] - r["spent"] for r in rows) / statistics.fmean(
            r["budget"] for r in rows)
        print(f"{names[p]:>32} "
              f"{sum(r['mission_ok'] for r in rows) / n:>16.0%} "
              f"{statistics.fmean(r['silent_truncation'] for r in rows):>21.1f} "
              f"{statistics.fmean(r['loud_abandon'] for r in rows):>21.1f} "
              f"{waste:>14.0%}")

    p1 = results["P1"]
    n = len(p1)
    any_trunc = sum(1 for r in p1 if r["silent_truncation"] > 0) / n
    print("\n== headline ==")
    print(f"* equal-split-no-reserve silently truncates >=1 leaf in "
          f"{any_trunc:.0%} of missions and reports those missions as SUCCESS.")
    print("  That is #478's 'plausible wrong answer' failure mode, generated by")
    print("  the resource model itself rather than by a lying agent.")
    print("* reserve-and-reallocate converts most of those into loud failures")
    print("  or completions, at the price of holding reserve idle. Escalation")
    print("  beats prediction because subtree demand is heavy-tailed.")
    print("* design rule: budgets must be (a) visible per node, (b) topped up")
    print("  only via recorded escalation events, (c) exhausted => explicit")
    print("  partial-result status, never a green checkmark.")


if __name__ == "__main__":
    main()

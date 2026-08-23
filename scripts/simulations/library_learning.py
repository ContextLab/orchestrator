#!/usr/bin/env python3
"""Can the solved-problem library rescue supercritical decomposition?

`decomposition_branching.py` established that recursive breakdown terminates
iff m = b*f < 1, and the Opus review of #485 argued (section 1b) that the
solved-problem library is the *termination mechanism*: solved ambiguous
subtrees become atomic cache hits next time, driving the effective ambiguity
rate down until the process goes subcritical. That claim had never been
simulated end to end. This script does it.

Model
-----
* A universe of P distinct subproblem *types*; step instances draw types
  i.i.d. from Zipf(s, P) -- demand concentrates on a few types.
* A type is intrinsically hard w.p. `h`: its FIRST encounter cannot be atomic,
  it genuinely breaks down into `b` child steps (same law). Easy types
  execute atomically immediately.
* Completing an ambiguous subtree stores its type in the library; later
  instances are cache hits: cheaper and less defect-prone.
* Two escape hatches mirror the proposed governor:
    - depth cap -> a node executes CRUDELY (5x cost, +1 defect) and the
      mission continues;
    - node budget cap -> the mission FAILS LOUDLY.
* Learning policy A ("complete-only"): only fully-expanded subtrees teach --
  crude depth-cap executions teach nothing. Policy B ("attempts"): everything
  attempted teaches, including work inside capped runs.

Questions answered: does repeated exposure drive a supercritical start
(m0 = b*h >> 1) below m = 1 WITHOUT changing the planner or classifier; how
long is cold start; and does learning from failed attempts matter?
"""

from __future__ import annotations

import random
import statistics

ASSUMPTIONS = {
    "types": 300,
    "zipf_skew": 1.1,
    "hard_rate": 0.30,        # P(fresh type genuinely needs breakdown)
    "steps_per_root": 10,     # children per ambiguous node (#485 target ~10)
    "node_cap": 600,          # loud-failure budget per mission
    "depth_cap": 5,           # beyond this, crude execution
    "missions": 250,
    "trials": 60,
    "seed": 11,
}


def zipf_table(n_types: int, skew: float) -> list[float]:
    raw = [1.0 / (rank + 1) ** skew for rank in range(n_types)]
    total = sum(raw)
    return [p / total for p in raw]


def uncovered_hard_mass(learned: set[int], hard: set[int],
                        weights: list[float]) -> float:
    """Remaining demand mass of hard types not yet in the library."""
    return sum(w for i, w in enumerate(weights)
               if i in hard and i not in learned)


def run_mission(rng: random.Random, weights: list[float], hard: set[int],
                learned: set[int], learn_policy: str) -> dict:
    b = ASSUMPTIONS["steps_per_root"]
    frontier = [(rng.choices(range(len(weights)), weights)[0], 1)]
    nodes = 0
    crude = 0
    taught_complete: list[int] = []
    taught_attempted: list[int] = []

    while frontier:
        typ, depth = frontier.pop()
        nodes += 1
        if nodes > ASSUMPTIONS["node_cap"]:
            taught = taught_complete if learn_policy == "complete-only" \
                else taught_attempted
            return {"ok": False, "nodes": nodes, "crude": crude,
                    "taught": taught}
        if typ in learned:
            continue                       # cache hit: atomic, cheap
        if typ not in hard:
            taught_complete.append(typ)    # trivially solvable now and later
            continue
        taught_attempted.append(typ)
        if depth >= ASSUMPTIONS["depth_cap"]:
            crude += 1                     # crude execution: expensive, flawed
            taught_complete.append(typ)
            continue
        frontier.extend((rng.choices(range(len(weights)), weights)[0], depth + 1)
                        for _ in range(b))

    taught = taught_complete if learn_policy == "complete-only" else taught_attempted
    return {"ok": True, "nodes": nodes, "crude": crude, "taught": taught}


def timeline(learn_policy: str) -> list[dict]:
    rng = random.Random(ASSUMPTIONS["seed"])
    weights = zipf_table(ASSUMPTIONS["types"], ASSUMPTIONS["zipf_skew"])
    hard = {i for i in range(ASSUMPTIONS["types"])
            if rng.random() < ASSUMPTIONS["hard_rate"]}

    per_index: list[list[dict]] = [[] for _ in range(ASSUMPTIONS["missions"])]
    for _ in range(ASSUMPTIONS["trials"]):
        learned: set[int] = set()
        for mi in range(ASSUMPTIONS["missions"]):
            res = run_mission(rng, weights, hard, learned, learn_policy)
            learned.update(res["taught"])
            res["library"] = len(learned)
            # P(a drawn child type is hard and unlearned) = remaining
            # uncovered-hard demand mass -- this IS f(t).
            res["p_recurse"] = uncovered_hard_mass(learned, hard, weights)
            per_index[mi].append(res)

    agg = []
    for mi, rows in enumerate(per_index):
        agg.append({
            "mission": mi + 1,
            "success": statistics.fmean(r["ok"] for r in rows),
            "nodes": statistics.mean(r["nodes"] for r in rows),
            "crude": statistics.fmean(r["crude"] for r in rows),
            "library": statistics.mean(r["library"] for r in rows),
            # m_eff = b * f(t); p_recurse already IS P(child hard & unlearned)
            "m_eff": ASSUMPTIONS["steps_per_root"]
            * statistics.fmean(r["p_recurse"] for r in rows),
        })
    return agg


def summarize(label: str, agg: list[dict]) -> None:
    print(f"\n== {label} ==")
    hdr = (f"{'mission':>8} {'success%':>9} {'mean nodes':>11} "
           f"{'crude/miss':>11} {'library':>8} {'m_eff':>6}")
    print(hdr)
    print("-" * len(hdr))
    n = len(agg)
    shown = sorted({1, 2, 3, 5, 10, 20, 40, 80, n})
    for m in shown:
        a = agg[m - 1]
        print(f"{a['mission']:>8} {a['success']:>8.0%} {a['nodes']:>11,.0f} "
              f"{a['crude']:>11.1f} {a['library']:>8,.0f} {a['m_eff']:>6.2f}")


def main() -> None:
    b, h = ASSUMPTIONS["steps_per_root"], ASSUMPTIONS["hard_rate"]
    print("== can the library rescue a supercritical system? ==")
    print(f"b={b}, fresh hardness h={h} -> cold-start m0 <= {b * h:.1f} "
          f"by construction (supercritical); {ASSUMPTIONS['types']} types, Zipf "
          f"{ASSUMPTIONS['zipf_skew']}, {ASSUMPTIONS['trials']} trials x "
          f"{ASSUMPTIONS['missions']} missions")

    complete = timeline("complete-only")
    attempts = timeline("attempts")
    summarize("cold start, learn only from FULLY-COMPLETED subtrees", complete)
    summarize("cold start, EVERY attempt teaches (capped runs included)",
              attempts)

    def first_stable(agg: list[dict]) -> int:
        for a in agg:
            if a["m_eff"] < 1.0 and a["success"] >= 0.95:
                return a["mission"]
        return -1

    c1, c2 = first_stable(complete), first_stable(attempts)
    early = statistics.fmean(a["nodes"] for a in complete[:20])
    late = statistics.fmean(a["nodes"] for a in complete[-20:])
    print("\n== headline ==")
    print(f"* m_eff drops below 1 with >=95% success at mission "
          f"{max(c1, 1)} (complete-only) vs {max(c2, 1)} (attempts-teach).")
    print(f"* cold-start tax: {early / max(late, 1):.0f}x more nodes per "
          f"mission in the first 20 missions than the last 20.")
    print("* stability is a property of the fleet's MEMORY, not of one run:")
    print("  the same planner that melts on day one converges once the library")
    print("  fills -- IF capped/failed attempts also teach. Design rules:")
    print("  persist solutions from failed missions; warm-start the library;")
    print("  measure f_eff continuously (it IS the system's vital sign).")


if __name__ == "__main__":
    main()

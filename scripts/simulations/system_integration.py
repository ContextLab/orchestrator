#!/usr/bin/env python3
"""The whole #485 proposal as one runnable schematic -- then ablated.

Component 1c of #485 says: turn the proposed solution into deterministic
operations over objects of the assumed shape, write predictions down, run,
and see what holds. Prior scripts test mechanisms in isolation; this one
assembles them into a mission-level Monte Carlo:

  * branching decomposition   (b steps/pipeline, ambiguity rate f0)
  * solved-problem library    (Zipf demand pool; cache hits are cheaper and
                               less defective; failed attempts teach)
  * review gates              (none / declarative-loop / evidential)
  * budget governor           (global token budget; exhaustion escalates,
                               never silently succeeds)

and then ablates subsystems to answer "what works and what doesn't":

  FULL      library + evidential review + budgets
  -LIBRARY  no cross-mission caching
  -REVIEW   outputs ship unchecked
  -BUDGETS  depth cap only (silent truncation possible)
  NAIVE     #485 as literally read: declarative unbounded review, depth cap,
            no budgets, library resets every mission

Outputs: success-within-budget rate, defective-shipped rate (declared success
but flawed result), mean token spend -- plus a scan of FULL over f0 x budget.
"""

from __future__ import annotations

import math
import random
import statistics

ASSUMPTIONS = {
    "types": 150,
    "zipf_skew": 1.2,
    "hard_rate": 0.35,
    "steps_per_root": 8,
    "fanout": 4,
    "missions_per_fleet": 40,
    "fleets": 24,
    "node_cost_atomic": 2_000,
    "node_cost_cached": 800,
    "decomp_cost": 1_500,
    "review_tokens_round": 8_000,
    "p_detect": 0.6,
    "lambda_fp_prose": 1.0,
    "fp_evidence_ratio": 0.20,
    "lambda_drift": 0.5,
    "round_cap": 12,
    "defect_rate_novel": 0.30,
    "defect_rate_cached": 0.08,
    "budget_multiple": 2.0,
    "depth_cap_naive": 8,
    "seed": 71,
}


def poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def binom(n: int, p: float, rng: random.Random) -> int:
    return sum(1 for _ in range(n) if rng.random() < p)


class Fleet:
    """One replicated world: demand law, library, review mode, budget mode."""

    def __init__(self, rng: random.Random, use_library: bool, review: str,
                 use_budgets: bool):
        self.rng = rng
        self.use_library = use_library
        self.review = review
        self.use_budgets = use_budgets
        n = ASSUMPTIONS["types"]
        raw = [1 / (r + 1) ** ASSUMPTIONS["zipf_skew"] for r in range(n)]
        total = sum(raw)
        self.weights = [p / total for p in raw]
        self.hard = {i for i in range(n)
                     if rng.random() < ASSUMPTIONS["hard_rate"]}
        self.library: set[int] = set()

    def draw_type(self) -> int:
        return self.rng.choices(range(len(self.weights)), self.weights)[0]

    def review_artifact(self, defects: int) -> tuple[int, int]:
        """Configured gate on one node artifact -> (residual defects, tokens)."""
        if self.review == "none":
            return defects, 0
        lam_ev = ASSUMPTIONS["lambda_fp_prose"] * ASSUMPTIONS["fp_evidence_ratio"]
        open_fps = 0
        rnd = 0
        for rnd in range(1, ASSUMPTIONS["round_cap"] + 1):
            found = binom(defects, ASSUMPTIONS["p_detect"], self.rng)
            if self.review == "evidential":
                found = round(found * 0.85)
                fps = poisson(self.rng, lam_ev)
                drift = 0
            else:
                fps = poisson(self.rng, ASSUMPTIONS["lambda_fp_prose"])
                drift = (poisson(self.rng, ASSUMPTIONS["lambda_drift"])
                         if self.review == "declarative" else 0)
            if found + fps + drift == 0 and open_fps == 0:
                return defects, rnd * ASSUMPTIONS["review_tokens_round"]
            defects -= found
            defects += binom(found, 0.15, self.rng)
            churn = fps + drift
            open_fps += churn
            defects += binom(churn, 0.15, self.rng)
            open_fps -= binom(open_fps, 0.85, self.rng)
        return defects, rnd * ASSUMPTIONS["review_tokens_round"]

    def run_mission(self) -> dict:
        """One mission through decomposition + execution + gates."""
        b = ASSUMPTIONS["steps_per_root"]
        budget = None
        reserve = 0.0
        if self.use_budgets:
            expected_nodes = 40
            budget = ASSUMPTIONS["budget_multiple"] * expected_nodes * (
                ASSUMPTIONS["node_cost_atomic"] * 0.7 +
                ASSUMPTIONS["review_tokens_round"] * 1.2)
            reserve = budget * 0.3
            budget -= reserve

        touched: list[int] = []
        spent = 0.0
        residual_defects = 0
        nodes = 0
        exhausted = False
        frontier = [(self.draw_type(), 1)]
        while frontier:
            typ, depth = frontier.pop()
            nodes += 1
            if not self.use_budgets and depth > ASSUMPTIONS["depth_cap_naive"]:
                residual_defects += 1     # silent truncation ships a flaw
                continue
            if typ in self.library:
                cost = ASSUMPTIONS["node_cost_cached"]
                defects = 1 if self.rng.random() < ASSUMPTIONS["defect_rate_cached"] else 0
            else:
                cost = ASSUMPTIONS["node_cost_atomic"]
                defects = 1 if self.rng.random() < ASSUMPTIONS["defect_rate_novel"] else 0
            res_def, rev_tok = self.review_artifact(defects)
            step_cost = cost + rev_tok
            if self.use_budgets:
                if step_cost > budget:
                    topup = min(step_cost - budget, reserve)
                    reserve -= topup
                    budget += topup
                    if step_cost > budget:
                        exhausted = True
                        break
                budget -= step_cost
            spent += step_cost
            residual_defects += res_def
            if typ not in self.library:
                if typ in self.hard and depth < 6:
                    touched.append(typ)
                    frontier.extend((self.draw_type(), depth + 1)
                                    for _ in range(ASSUMPTIONS["fanout"]))
        if self.use_library:
            self.library.update(touched)   # failed attempts teach too

        return {
            "ok": not exhausted,
            "spent": spent,
            "residual": residual_defects,
            "nodes": nodes,
            "library": len(self.library),
        }


def run_config(name: str, use_library: bool, review: str,
               use_budgets: bool) -> dict:
    # str hash() is process-randomized; derive the seed deterministically
    rng = random.Random(ASSUMPTIONS["seed"] + sum(map(ord, name)))
    rows = []
    for _ in range(ASSUMPTIONS["fleets"]):
        fleet = Fleet(rng, use_library, review, use_budgets)
        for _mi in range(ASSUMPTIONS["missions_per_fleet"]):
            rows.append(fleet.run_mission())
    n = len(rows)
    ok_rows = [r for r in rows if r["ok"]]
    return {
        "name": name,
        "success": len(ok_rows) / n,
        "defective_of_declared": shipped_defected_rate(ok_rows),
        "mean_spend": statistics.fmean(r["spent"] for r in rows),
        "_n": n,
    }


def shipped_defected_rate(ok_rows: list[dict]) -> float:
    if not ok_rows:
        return float("nan")
    return sum(1 for r in ok_rows if r["residual"] > 0) / len(ok_rows)


def ablation_table() -> None:
    configs = [
        ("FULL", True, "evidential", True),
        ("-LIBRARY", False, "evidential", True),
        ("-REVIEW", True, "none", True),
        ("-BUDGETS", True, "evidential", False),
        ("NAIVE", False, "declarative", False),
    ]
    hdr = (f"{'config':>10} {'success%':>9} {'defect%|declared':>17} "
           f"{'mean spend':>12} {'missions':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, lib, rev, bud in configs:
        r = run_config(name, lib, rev, bud)
        print(f"{name:>10} {r['success']:>8.0%} {r['defective_of_declared']:>16.0%} "
              f"{r['mean_spend']:>12,.0f} {r['_n']:>9,}")


def phase_scan() -> None:
    saved = {k: v for k, v in ASSUMPTIONS.items()}
    print("\n== FULL config: success%% over hardness x budget ==")
    grid_hard = [0.20, 0.35, 0.50]
    grid_bud = [0.25, 0.5, 1.0]   # tight budgets: where the governor binds
    row_label = "hard vs budget"
    hdr = f"{row_label:>14}" + "".join(f"{b:>9.1f}x" for b in grid_bud)
    print(hdr)
    for h in grid_hard:
        ASSUMPTIONS["hard_rate"] = h
        cells = []
        for bm in grid_bud:
            ASSUMPTIONS["budget_multiple"] = bm
            rng = random.Random(ASSUMPTIONS["seed"] + int(h * 100) + int(bm * 10))
            outcomes = []
            for _ in range(ASSUMPTIONS["fleets"] // 2):
                fleet = Fleet(rng, True, "evidential", True)
                for _mi in range(ASSUMPTIONS["missions_per_fleet"]):
                    outcomes.append(fleet.run_mission()["ok"])
            cells.append(statistics.fmean(outcomes))
        print(f"{h:>13.2f}" + "".join(f"{c:>8.0%}" for c in cells))
    ASSUMPTIONS.clear()
    ASSUMPTIONS.update(saved)


def main() -> None:
    print("== integrated system Monte Carlo (component 1c applied to #485) ==")
    print(f"b={ASSUMPTIONS['steps_per_root']}, fresh-type hardness "
          f"f0={ASSUMPTIONS['hard_rate']} -> m0={ASSUMPTIONS['steps_per_root'] * ASSUMPTIONS['hard_rate']:.1f}; "
          f"novel-defect rate {ASSUMPTIONS['defect_rate_novel']}, "
          f"cached {ASSUMPTIONS['defect_rate_cached']}; "
          f"{ASSUMPTIONS['fleets']} fleets x {ASSUMPTIONS['missions_per_fleet']} missions/config")
    ablation_table()
    phase_scan()
    print("\n== reading ==")
    print("* Every ablation should HURT somewhere: library cuts spend over time,")
    print("  review cuts shipped defects, budgets convert silent truncation into")
    print("  loud failure. If removing one barely moves anything, that subsystem")
    print("  is not earning its complexity yet.")
    print("* NAIVE vs FULL is the whole argument of this issue thread in one row:")
    print("  the difference is not intelligence, it is governance.")


if __name__ == "__main__":
    main()

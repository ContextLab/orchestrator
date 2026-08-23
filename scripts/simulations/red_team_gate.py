#!/usr/bin/env python3
"""What does #485's red-team loop actually buy, and what does "clean" certify?

#485 requires that every artifact be red-teamed by a separate agent, looping
until the reviewer's concerns "come back clean", with concerns tracked so scope
does not drift. No authoring agent may review its own work, though the same
*model* may review another instance of itself in a fresh session.

Three questions, three models.

**(a) Is the concern ledger worth it?** Yes, and it is the whole mechanism.
Unbounded scope drift turns a 3.6-round loop into 8.6 rounds with a 2.2%
never-terminating tail; the ledger removes drift's effect entirely. Concretely:
freeze the acceptance criteria *before* authoring; every concern gets an id, a
state, and a link to the frozen criterion it violates; a concern that maps to
no criterion is auto-filed as deferred rather than blocking.

**(b) What does a clean verdict certify?** Much less than it looks like. At a
plausible 50% detection rate, only 45% of clean passes are actually clean --
a clean verdict is wrong more often than it is right. The loop is
self-deceiving in a specific way: it terminates exactly when the reviewer stops
finding things, which tracks reviewer fatigue as much as artifact quality.
The report's fix is to change what a reviewer may *submit*: a blocking concern
must ship a reproducible artifact (a failing assertion, a counterexample input,
a contradicting citation, a command whose output differs from the prediction).
Prose worries are recorded as non-blocking risks. That terminates on evidence,
makes ``p_detect`` measurable by counting artifacts, and composes with #485's
own component 1c.

**(c) How much independence does "same model, fresh session" buy?** Let ``rho``
be the probability that a defect is systematically invisible to a model family.
Same-family reviewers are conditionally independent only on the ``1-rho``
fraction, so no number of them beats a ``1-rho`` ceiling. One reviewer from a
different family beats five from the same one once ``rho > ~0.1``. This
collides with the two-provider policy in #430/#484.

**Assumptions** in ASSUMPTIONS. The structural conclusions -- the ledger makes
rounds independent of drift, a clean verdict is weak evidence at low
``p_detect``, and the ``1-rho`` ceiling -- hold for any values.
"""

from __future__ import annotations

import random
import statistics

# --- assumptions -----------------------------------------------------------
ASSUMPTIONS = {
    "initial_defects": 6,
    "p_detect": 0.6,  # P(reviewer finds a given defect in a round)
    "p_regress": 0.15,  # P(a fix introduces a new defect)
    "max_rounds": 25,
    "trials": 2_000,
    "seed": 11,
    "p_catch": 0.6,  # per-reviewer catch rate, for the independence model
}


def review_loop(
    initial_defects: int,
    p_detect: float,
    p_regress: float,
    p_drift: float,
    ledger: bool,
    rng: random.Random,
    max_rounds: int = ASSUMPTIONS["max_rounds"],
) -> tuple[int, int, bool]:
    """One author/red-team loop.

    Each round the reviewer finds each outstanding defect with probability
    ``p_detect``; each fix introduces a new defect with probability
    ``p_regress``; the reviewer also raises a new-*scope* concern with
    probability ``p_drift``. With ``ledger=True`` an out-of-scope concern is
    recorded and deferred rather than blocking the gate.

    Returns (rounds used, defects remaining, converged).
    """
    defects, rounds = initial_defects, 0
    while rounds < max_rounds:
        rounds += 1
        found = sum(1 for _ in range(defects) if rng.random() < p_detect)
        drifted = rng.random() < p_drift
        blocking = found + (0 if ledger else int(drifted))
        if blocking == 0:
            return rounds, defects, True
        defects -= found
        defects += sum(1 for _ in range(found) if rng.random() < p_regress)
        if not ledger and drifted:
            defects += 1  # a redefinition becomes real new work
    return rounds, defects, False


def loop_stats(p_drift: float, ledger: bool, p_detect: float = ASSUMPTIONS["p_detect"],
               trials: int = ASSUMPTIONS["trials"],
               seed: int = ASSUMPTIONS["seed"]) -> dict:
    """Aggregate ``trials`` review loops."""
    rng = random.Random(seed)
    runs = [
        review_loop(
            ASSUMPTIONS["initial_defects"], p_detect, ASSUMPTIONS["p_regress"],
            p_drift, ledger, rng,
        )
        for _ in range(trials)
    ]
    converged = [r for r in runs if r[2]]
    return {
        "mean_rounds": statistics.mean(r[0] for r in converged) if converged else float("nan"),
        "p95_rounds": sorted(r[0] for r in converged)[int(0.95 * len(converged))]
        if converged
        else float("nan"),
        "never_clean_pct": 100.0 * (1 - len(converged) / trials),
        "defects_left": statistics.mean(r[1] for r in runs),
        "p_truly_clean": (
            sum(1 for r in converged if r[1] == 0) / len(converged) if converged else float("nan")
        ),
        "residual_given_clean": (
            statistics.mean(r[1] for r in converged) if converged else float("nan")
        ),
    }


def catch_rate(rho: float, n_reviewers: int, p_catch: float = ASSUMPTIONS["p_catch"]) -> float:
    """P(defect caught) by ``n_reviewers`` from the SAME model family.

    A share ``rho`` of defects is invisible to the whole family; reviewers are
    conditionally independent only on the rest. Ceiling is ``1 - rho``.
    """
    return 1 - rho - (1 - rho) * (1 - p_catch) ** n_reviewers


def cross_family_catch_rate(rho: float, p_catch: float = ASSUMPTIONS["p_catch"]) -> float:
    """P(defect caught) by two reviewers from families with independent blind spots."""
    joint_blind = rho * rho
    return 1 - joint_blind - (1 - joint_blind) * (1 - p_catch) ** 2


def main() -> None:
    print("== rounds to a clean red-team pass ==")
    print(
        f"{ASSUMPTIONS['trials']:,} trials; D0={ASSUMPTIONS['initial_defects']} latent "
        f"defects, p_detect={ASSUMPTIONS['p_detect']}, "
        f"p_regress={ASSUMPTIONS['p_regress']}"
    )
    print()
    hdr = (
        f"{'p_drift':>8} {'ledger':>8} {'mean rounds':>12} {'p95 rounds':>11} "
        f"{'never clean':>12} {'defects left':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p_drift in (0.0, 0.2, 0.4, 0.6):
        for ledger in (False, True):
            s = loop_stats(p_drift, ledger)
            print(
                f"{p_drift:>8} {str(ledger):>8} {s['mean_rounds']:>12.1f} "
                f"{s['p95_rounds']:>11} {s['never_clean_pct']:>11.1f}% "
                f"{s['defects_left']:>13.2f}"
            )

    print()
    print("== what a clean pass actually certifies ==")
    print("P(zero defects remain | the reviewer returned clean), by detection rate")
    print()
    hdr2 = (
        f"{'p_detect':>9} {'mean rounds':>12} {'P(truly clean)':>15} "
        f"{'mean residual defects':>22}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for p_detect in (0.3, 0.5, 0.7, 0.9):
        s = loop_stats(0.0, True, p_detect=p_detect, trials=4_000, seed=3)
        print(
            f"{p_detect:>9} {s['mean_rounds']:>12.1f} {s['p_truly_clean']:>14.0%} "
            f"{s['residual_given_clean']:>22.2f}"
        )

    print()
    print("== reviewer independence ceiling ==")
    print("rho = P(a defect is systematically invisible to this model family).")
    print("Same-family reviewers are conditionally independent only on the 1-rho part.")
    print()
    hdr3 = (
        f"{'rho':>6} {'1 rev':>8} {'2 rev':>8} {'3 rev':>8} {'5 rev':>8} "
        f"{'inf rev':>9} {'+1 other family':>17}"
    )
    print(hdr3)
    print("-" * len(hdr3))
    for rho in (0.0, 0.1, 0.2, 0.3, 0.5):
        row = " ".join(f"{catch_rate(rho, n):>7.0%}" for n in (1, 2, 3, 5))
        print(
            f"{rho:>6} {row} {1 - rho:>8.0%} "
            f"{cross_family_catch_rate(rho):>16.0%}"
        )
    print()
    print("  No number of same-family reviewers beats the 1-rho ceiling.")
    print("  One reviewer from a different family beats five from the same one")
    print("  whenever rho > ~0.1.")


if __name__ == "__main__":
    main()

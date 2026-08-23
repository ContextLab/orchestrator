"""The #485 design simulations still demonstrate what the review said they did.

`scripts/simulations/` backs the design review on #485 with four models. Their
value is entirely in a handful of *structural* claims -- claims that hold for
any parameter values, and that the review leans on. If a refactor quietly
breaks one, the review's argument stops being reproducible and the scripts
become decoration.

So this file asserts the structure, not the tables. A golden-output test would
fail for every cosmetic change and pass for a silently wrong model, which is
the wrong way round. The specific numbers in those tables are only as good as
each script's `ASSUMPTIONS` dict, and are expected to move when someone
replaces a guess with a measurement -- see `scripts/simulations/README.md`.

Determinism is asserted separately: every stochastic model is seeded, so the
review's numbers can be reproduced exactly.
"""

import importlib.util
import random
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parent.parent
SIMULATIONS = REPO / "scripts" / "simulations"


def _module(name: str):
    """Load a simulation by path; `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, SIMULATIONS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


branching = _module("decomposition_branching")
scratchpad = _module("scratchpad_contention")
context = _module("context_recursion")
ASSUMPTIONS_DEPTH_CAP = context.ASSUMPTIONS["depth_cap"]
red_team = _module("red_team_gate")


# --- decomposition_branching ------------------------------------------------


@pytest.mark.parametrize(
    "b, f, expected",
    [
        (5, 0.05, "subcritical"),
        (10, 0.05, "subcritical"),
        (5, 0.20, "critical"),
        (10, 0.10, "critical"),
        (10, 0.30, "SUPERCRITICAL"),
    ],
)
def test_regime_is_decided_by_mean_offspring(b, f, expected):
    """m = b*f is the whole classification; nothing else enters into it."""
    assert branching.regime(b, f) == expected


def test_subcritical_trees_stay_small_and_supercritical_ones_explode():
    """The claim the review rests on: crossing m = 1 changes the outcome by orders."""
    sub = branching.leaf_distribution(10, 0.05)
    sup = branching.leaf_distribution(10, 0.30)
    assert sub["m"] < 1 < sup["m"]
    assert sub["median"] < 100
    assert sup["median"] > 10_000
    assert sup["median"] > 100 * sub["median"]


def test_leaf_count_rises_monotonically_with_ambiguity_rate():
    """More ambiguity is never cheaper. Guards against a sign or bounds slip."""
    medians = [branching.leaf_distribution(10, f)["median"] for f in (0.05, 0.1, 0.15, 0.2, 0.3)]
    assert medians == sorted(medians)


def test_branching_is_seeded_and_reproducible():
    a = branching.branch(10, 0.3, random.Random(7))
    b = branching.branch(10, 0.3, random.Random(7))
    assert a == b


def test_depth_cap_bounds_the_tree():
    """Without the cap, a supercritical config would not return at all."""
    _, _, max_depth = branching.branch(10, 0.9, random.Random(1), depth_cap=4)
    assert max_depth < 4


def test_most_of_a_node_budget_is_spent_before_any_work_happens():
    """The review's "34% shared-state re-reading, 8% work" point."""
    assert branching.shared_state_share() > 0.3
    work_share = 12_000 / branching.tokens_per_node()
    assert work_share < 0.1


# --- scratchpad_contention --------------------------------------------------


def test_locked_throughput_saturates_at_the_critical_section_ceiling():
    """Adding agents past saturation buys latency, not throughput."""
    ceiling = scratchpad.throughput_ceiling(8.0)
    saturated = [scratchpad.simulate_lock(n, 8.0, 60.0)["steps_per_hour"] for n in (16, 32, 64, 128)]
    assert ceiling == pytest.approx(450.0)
    for observed in saturated:
        assert observed <= ceiling * 1.05
        assert observed > ceiling * 0.9
    # 8x the fleet moves throughput by less than 5%: the definition of saturated.
    assert abs(saturated[-1] - saturated[0]) / saturated[0] < 0.05


def test_lock_wait_grows_with_fleet_size_once_saturated():
    waits = [scratchpad.simulate_lock(n, 8.0, 60.0)["mean_wait"] for n in (16, 32, 64, 128)]
    assert waits == sorted(waits)
    assert waits[-1] > 10 * waits[0] / 10  # strictly increasing, and by a lot
    assert waits[-1] > 600


def test_moving_the_model_call_out_of_the_lock_lifts_the_ceiling():
    """The review's headline fix, and the ~61x figure at 512 agents."""
    locked = scratchpad.throughput_ceiling(8.0)
    unlocked = scratchpad.simulate_lock(512, 0.03, 68.0)["steps_per_hour"]
    assert unlocked / locked > 50


def test_sealed_segments_turn_quadratic_churn_into_linear():
    """Naive re-summarisation grows quadratically; sealed segments do not."""
    small, large = scratchpad.churn(1_000), scratchpad.churn(100_000)
    naive_growth = large["naive"] / small["naive"]
    sealed_growth = large["sealed"] / small["sealed"]
    assert naive_growth > 1_000  # ~100x the notes -> ~10,000x the work
    assert sealed_growth < 200  # ~100x the notes -> ~100x the work
    assert large["naive"] / large["sealed"] > 10_000


def test_lock_simulation_is_seeded_and_reproducible():
    a = scratchpad.simulate_lock(32, 8.0, 60.0, seed=0)
    b = scratchpad.simulate_lock(32, 8.0, 60.0, seed=0)
    assert a == b


def test_lock_utilisation_never_exceeds_one():
    """An earlier draft counted service past the horizon and reported 127%."""
    for n in (2, 16, 128):
        assert scratchpad.simulate_lock(n, 8.0, 60.0)["utilisation"] <= 1.0


# --- context_recursion ------------------------------------------------------


def test_summary_tree_converges_for_any_real_compression_ratio():
    for r in (0.1, 0.5, 0.9):
        tree = context.summary_tree(5_000_000, 1_000_000, r=r)
        assert tree["converges"]
        assert tree["depth_required"] < float("inf")


def test_summary_tree_does_not_converge_when_summaries_do_not_shrink():
    """r >= 1 is the failure mode; it must be reported, not silently capped."""
    for r in (1.0, 1.1):
        tree = context.summary_tree(5_000_000, 1_000_000, r=r)
        assert not tree["converges"]
        assert tree["depth_required"] == float("inf")


def test_converging_and_fitting_within_the_depth_cap_are_different_claims():
    """r=0.9 terminates mathematically but needs 29 levels. Do not conflate them."""
    gentle = context.summary_tree(5_000_000, 1_000_000, r=0.9)
    assert gentle["converges"]
    assert gentle["hit_cap"]
    assert gentle["depth_required"] > ASSUMPTIONS_DEPTH_CAP

    aggressive = context.summary_tree(5_000_000, 1_000_000, r=0.1)
    assert aggressive["converges"]
    assert not aggressive["hit_cap"]
    assert aggressive["top"] <= aggressive["payload"]


def test_top_level_fidelity_is_set_by_target_size_not_by_compression_ratio():
    """Fidelity ~ payload/N whatever r is; r only buys you fewer lossy hops."""
    n_tokens, ctx = 5_000_000, 1_000_000
    fidelities = []
    for r in (0.1, 0.5, 0.9):
        tree = context.summary_tree(n_tokens, ctx, r=r)
        fidelities.append(r ** tree["depth_required"])
    forced = context.summary_tree(n_tokens, ctx)["payload"] / n_tokens
    for observed in fidelities:
        assert observed <= forced * 1.05
        assert observed > forced / 10
    # But the hop count -- and so the accumulated distortion -- differs hugely.
    hops = [context.summary_tree(n_tokens, ctx, r=r)["depth_required"] for r in (0.1, 0.9)]
    assert hops[1] > 10 * hops[0]


def test_depth_is_logarithmic_not_linear_in_document_size():
    """400x the corpus must not cost 400x the depth."""
    small = context.summary_tree(250_000, 1_000_000)["depth"]
    large = context.summary_tree(100_000_000, 1_000_000)["depth"]
    assert large <= small + 5
    assert large <= 5


def test_total_cost_is_a_small_multiple_of_the_corpus():
    tree = context.summary_tree(100_000_000, 1_000_000)
    assert tree["tokens"] < 2.5 * 100_000_000


def test_fidelity_decays_geometrically_with_depth():
    """Why the summary tree cannot be the retrieval path."""
    tree = context.summary_tree(100_000_000, 128_000, r=0.2)
    assert tree["fidelity"] == pytest.approx(0.2 ** tree["depth"])
    assert tree["fidelity"] < 0.001


def test_overheads_shrink_the_usable_chunk_below_the_nominal_payload():
    """Ignoring instructions + output reserve overstates capacity."""
    tree = context.summary_tree(1, 1_000_000, r=0.4)
    assert tree["chunk"] < tree["payload"]
    assert tree["chunk"] > 0.7 * tree["payload"]


def test_impossible_overheads_raise_rather_than_loop():
    with pytest.raises(ValueError):
        context.summary_tree(1_000_000, 1_000, instr=100_000)


def test_specified_budget_consumes_half_the_context_before_any_work():
    assert context.budget_subtotal() == pytest.approx(0.50)


# --- red_team_gate ----------------------------------------------------------


def test_the_ledger_makes_review_rounds_independent_of_scope_drift():
    """The concern ledger's entire purpose, and it delivers exactly that."""
    with_ledger = [red_team.loop_stats(d, ledger=True)["mean_rounds"] for d in (0.0, 0.2, 0.4, 0.6)]
    assert with_ledger == pytest.approx([with_ledger[0]] * 4)


def test_without_a_ledger_drift_inflates_rounds_and_leaves_a_hung_tail():
    baseline = red_team.loop_stats(0.0, ledger=False)
    drifting = red_team.loop_stats(0.6, ledger=False)
    assert drifting["mean_rounds"] > 2 * baseline["mean_rounds"]
    assert drifting["never_clean_pct"] > 1.0
    assert red_team.loop_stats(0.6, ledger=True)["never_clean_pct"] == 0.0


def test_a_clean_verdict_is_weak_evidence_at_a_realistic_detection_rate():
    """The uncomfortable result: at p_detect=0.5, "clean" is wrong more often than right."""
    low = red_team.loop_stats(0.0, True, p_detect=0.5, trials=4_000, seed=3)
    assert low["p_truly_clean"] < 0.5
    assert low["residual_given_clean"] > 0.5


def test_certification_strength_rises_with_detection_rate():
    rates = [
        red_team.loop_stats(0.0, True, p_detect=p, trials=4_000, seed=3)["p_truly_clean"]
        for p in (0.3, 0.5, 0.7, 0.9)
    ]
    assert rates == sorted(rates)
    assert rates[0] < 0.2 and rates[-1] > 0.9


def test_same_family_reviewers_cannot_beat_the_blind_spot_ceiling():
    """No amount of "fresh session, same model" review passes 1 - rho."""
    for rho in (0.1, 0.3, 0.5):
        assert red_team.catch_rate(rho, 1_000) <= 1 - rho + 1e-9
        assert red_team.catch_rate(rho, 5) < 1 - rho


def test_one_cross_family_reviewer_beats_five_same_family_ones_once_rho_is_real():
    assert red_team.cross_family_catch_rate(0.0) < red_team.catch_rate(0.0, 5)
    for rho in (0.2, 0.3, 0.5):
        assert red_team.cross_family_catch_rate(rho) > red_team.catch_rate(rho, 5)


def test_review_loop_is_seeded_and_reproducible():
    a = red_team.review_loop(6, 0.6, 0.15, 0.4, False, random.Random(11))
    b = red_team.review_loop(6, 0.6, 0.15, 0.4, False, random.Random(11))
    assert a == b

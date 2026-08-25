"""Derivation tests for :mod:`sherpa.metrics` (issue #492 §3/§6).

These tests exist because the project's central viability gate --- "corrected
``m = E[ambiguous children per decomposition]`` has an upper confidence bound
below 1" --- was previously computed from numbers that could not possibly carry
the signal it claims to carry:

D1  ``_decompose_stats`` counted ``not c.atomic_claim``, i.e. what the PLANNER
    DECLARED, and never read a single ``admission_checked`` event.  #492 §3 is
    explicit that "an ``atomic`` step is an admitted executable claim, not a
    planner label"; a metric blind to admission is a self-report.

D2  The kernel emits ``decompose_outcome`` only when the parent node is not
    ``skipped``, and the reclassify path sets the parent to ``skipped`` --- so a
    reclassified decomposition emitted NO event, and the corrected branching
    factor structurally excluded the only correction signal that exists.  All
    41 ``m_values`` in ``benchmarks/artifacts/suite.json`` were exactly 0.0.

D3  Existing coverage only asserted ``m_corrected < 1.0``, which an
    identically-zero metric satisfies trivially.  Every arithmetic assertion
    below is an EXACT hand-computed value, worked out in the docstring.

D4  ``bootstrap_ci`` was exercised only with the constant list ``[0.4] * 20``,
    where every resample is identical and therefore ANY implementation passes
    --- including ``return (min(values), max(values))``.

D5  Usage totals initialised to ``0.0`` reported "no model was ever consulted"
    identically to "the model reported zero tokens".

Everything here runs against a REAL :class:`sherpa.store.Store` (SQLite in WAL
mode) holding REAL :class:`sherpa.events.Event` rows.  No mocks anywhere: the
event log is the contract under test, so it must be a real event log.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from sherpa.events import Event
from sherpa.metrics import aggregate_run_reports, bootstrap_ci, render_report_md, run_metrics

RUN = "r-derivation"


# --------------------------------------------------------------------------- helpers


def _log(store, kind: str, *, node_key: str | None = None, **payload) -> None:
    """Append one REAL event to the REAL store (no mocks, no fixtures-in-memory)."""
    store.append(Event(kind=kind, run_id=RUN, node_key=node_key, payload=payload))


def _decompose(store, node_key: str, declared: int, ambiguous: int) -> None:
    """One measured decomposition: parent ``node_key`` fanned out to ``declared``."""
    _log(store, "decompose_outcome", node_key=node_key,
         children_declared=declared, children_ambiguous=ambiguous)


def _admission(store, node_key: str, *, claimed: bool, decision: str,
               capability: str = "fs.read_file") -> None:
    """One admission verdict for one node, in the kernel's own payload shape."""
    _log(store, "admission_checked", node_key=node_key, capability=capability,
         decision=decision, atomic_claimed=claimed, io_compatible=True,
         probe_ok=decision == "admitted", evidence_sha=None, reasons=[])


def _metrics(store) -> dict:
    return run_metrics(store.events(run_id=RUN))


# --------------------------------------------------------------------------- D1/D2/D3


class TestCorrectedBranchingIsDerivedFromAdmissionEvents:
    """D1: the corrected fan-out must read ``admission_checked``, not planner labels."""

    def test_reclassified_atomic_claim_counts_as_an_ambiguous_child(self, store) -> None:
        """One decomposition, 4 declared children, planner says all 4 are atomic.

        Admission disagrees about exactly one of them: ``c3``'s probe failed and
        it was sent back for decomposition.  By hand:

            D  = 1 decomposition
            C  = 4 declared children
            A_d= 0 planner-declared ambiguous children
            R  = 1 atomic claim reclassified by admission
            E  = 0 escalations
            V  = C - E                    = 4 viable children
            A  = A_d + R - E_ambiguous    = 0 + 1 - 0 = 1 corrected ambiguous child

            b_declared  = C / D = 4.0
            f_declared  = A_d / C = 0.0        <- the self-report: "no ambiguity"
            b_corrected = V / D = 4.0
            f_ambiguous = A / V = 0.25
            m_corrected = A / D = 1.0

        The planner-label metric says f = 0.0 and therefore m = 0.0 ("perfectly
        subcritical"); the measured one says m = 1.0, i.e. exactly critical.
        That gap is the whole point of admission control.
        """
        _decompose(store, "p0", declared=4, ambiguous=0)
        _admission(store, "c1", claimed=True, decision="admitted")
        _admission(store, "c2", claimed=True, decision="admitted")
        _admission(store, "c3", claimed=True, decision="reclassify_decompose")
        _admission(store, "c4", claimed=True, decision="admitted")

        br = _metrics(store)["branching"]
        assert br["decompositions"] == 1, br
        assert br["children_declared"] == 4, br
        assert br["children_ambiguous_declared"] == 0, br
        assert br["children_reclassified"] == 1, br
        assert br["children_ambiguous_corrected"] == 1, br
        assert br["b_declared"] == 4.0, br
        assert br["f_declared"] == 0.0, br
        assert br["b_corrected"] == 4.0, br
        assert br["f_ambiguous"] == 0.25, br
        assert br["m_corrected"] == 1.0, br

    def test_declared_figures_stay_planner_labels(self, store) -> None:
        """``b_declared``/``f_declared`` are legitimately self-reports and must not move.

        D = 2, C = 3 + 2 = 5, A_d = 1 + 0 = 1.
        b_declared = 5/2 = 2.5 ; f_declared = 1/5 = 0.2 --- regardless of the
        two reclassifications admission recorded.
        """
        _decompose(store, "p0", declared=3, ambiguous=1)
        _decompose(store, "p1", declared=2, ambiguous=0)
        _admission(store, "c1", claimed=True, decision="reclassify_decompose")
        _admission(store, "c2", claimed=True, decision="reclassify_decompose")

        br = _metrics(store)["branching"]
        assert br["b_declared"] == 2.5, br
        assert br["f_declared"] == 0.2, br
        # ... while the corrected figures DID move: A = 1 + 2 - 0 = 3, V = 5.
        assert br["children_ambiguous_corrected"] == 3, br
        assert br["f_ambiguous"] == 0.6, br
        assert br["m_corrected"] == 1.5, br

    def test_escalated_children_are_not_viable_children(self, store) -> None:
        """An escalated step is a refused proposal, not a viable child.

        D = 1, C = 4, A_d = 1.  Admission escalates two of them: ``c4`` which the
        planner had declared ambiguous (so it is already inside A_d), and ``c3``
        which the planner had claimed atomic.

            E   = 2, E_ambiguous = 1
            V   = 4 - 2 = 2
            A   = A_d + R - E_ambiguous = 1 + 0 - 1 = 0
            b_corrected = 2 / 1 = 2.0
            f_ambiguous = 0 / 2 = 0.0
            m_corrected = 0 / 1 = 0.0   (a MEASURED zero, see the D2 test below)
        """
        _decompose(store, "p0", declared=4, ambiguous=1)
        _admission(store, "c1", claimed=True, decision="admitted")
        _admission(store, "c2", claimed=True, decision="admitted")
        _admission(store, "c3", claimed=True, decision="escalate")
        _admission(store, "c4", claimed=False, decision="escalate")

        br = _metrics(store)["branching"]
        assert br["children_escalated"] == 2, br
        assert br["children_viable"] == 2, br
        assert br["b_declared"] == 4.0, br
        assert br["b_corrected"] == 2.0, br
        assert br["f_ambiguous"] == 0.0, br
        assert br["m_corrected"] == 0.0, br

    def test_repeated_admission_checks_on_one_node_count_once(self, store) -> None:
        """A node re-checked under the same key has ONE final disposition.

        ``c1`` is checked twice: first reclassified, then (after the operator
        widened authority) admitted.  Counting events rather than nodes would
        record a phantom extra ambiguous child.

        D = 1, C = 2, A_d = 0, R = 0 (c1's LAST verdict is ``admitted``),
        E = 0, V = 2, A = 0, m_corrected = 0.0.
        """
        _decompose(store, "p0", declared=2, ambiguous=0)
        _admission(store, "c1", claimed=True, decision="reclassify_decompose")
        _admission(store, "c1", claimed=True, decision="admitted")
        _admission(store, "c2", claimed=True, decision="admitted")

        br = _metrics(store)["branching"]
        assert br["children_reclassified"] == 0, br
        assert br["m_corrected"] == 0.0, br
        # The raw event tally is still honest about how many checks ran.
        assert _metrics(store)["admission"]["checked"] == 3


class TestReclassifyWithoutDecomposeOutcome:
    """D2: the kernel's ``skipped`` parent swallowed the reclassify decomposition."""

    def test_reclassify_with_no_decompose_outcome_is_not_silently_zero(self, store) -> None:
        """Reproduces the measured pre-fix tree exactly.

        A probe-failing capability produced::

            admission:  {overclaim_rate: 0.5, decisions: {reclassify_decompose: 1,
                                                          admitted: 2}}
            branching:  {decompositions: 0, f_ambiguous: 0.0, b_corrected: 0.0,
                         m_corrected: 0.0}

        with ZERO ``decompose_outcome`` events logged.  Reporting 0.0 there is a
        lie: no fan-out was observed at all, so the correct answer is "unknown"
        (``None``), plus a loud count of the decompositions we KNOW happened but
        whose fan-out never reached the log.
        """
        _admission(store, "c1", claimed=True, decision="reclassify_decompose")
        _admission(store, "c2", claimed=True, decision="admitted")
        _admission(store, "c3", claimed=False, decision="admitted")

        m = _metrics(store)
        assert m["admission"]["overclaim_rate"] == 0.5, m["admission"]
        assert m["admission"]["decisions"] == {"reclassify_decompose": 1, "admitted": 2}

        br = m["branching"]
        assert br["decompositions"] == 0, br
        assert br["decompositions_unmeasured"] == 1, br
        assert br["b_declared"] is None, br
        assert br["b_corrected"] is None, br
        assert br["f_ambiguous"] is None, br
        assert br["m_corrected"] is None, br

    def test_measured_zero_is_distinguishable_from_unknown(self, store) -> None:
        """All-atomic decomposition: m is a MEASURED 0.0, not a missing number.

        D = 1, C = 3, A_d = 0, R = 0, E = 0 -> V = 3, A = 0, m = 0/1 = 0.0.
        """
        _decompose(store, "p0", declared=3, ambiguous=0)
        _admission(store, "c1", claimed=True, decision="admitted")
        _admission(store, "c2", claimed=True, decision="admitted")
        _admission(store, "c3", claimed=True, decision="admitted")

        br = _metrics(store)["branching"]
        assert br["m_corrected"] == 0.0, br
        assert br["m_corrected"] is not None
        assert br["decompositions_unmeasured"] == 0, br

    def test_partially_measured_run_reports_the_gap(self, store) -> None:
        """``c1`` reclassified WITH its fan-out logged; ``c9`` reclassified without.

        D = 1 measured (node ``c1``), unmeasured = 1 (node ``c9``).
        C = 2, A_d = 0, R = 2, E = 0, V = 2, A = 2.
        b_corrected = 2/1 = 2.0 ; f_ambiguous = 2/2 = 1.0 ; m_corrected = 2/1 = 2.0.
        """
        _decompose(store, "c1", declared=2, ambiguous=0)
        _admission(store, "c1", claimed=True, decision="reclassify_decompose")
        _admission(store, "c9", claimed=True, decision="reclassify_decompose")

        br = _metrics(store)["branching"]
        assert br["decompositions"] == 1, br
        assert br["decompositions_unmeasured"] == 1, br
        assert br["b_corrected"] == 2.0, br
        assert br["f_ambiguous"] == 1.0, br
        assert br["m_corrected"] == 2.0, br


class TestBranchingArithmetic:
    """D3: exact hand-computed values, including every degenerate denominator."""

    def test_no_events_at_all(self, store) -> None:
        """Zero denominators everywhere -> None, never a fabricated 0.0."""
        _log(store, "run_started")
        m = _metrics(store)
        br = m["branching"]
        assert br["decompositions"] == 0
        assert br["children_declared"] == 0
        assert (br["b_declared"], br["f_declared"]) == (None, None)
        assert (br["b_corrected"], br["f_ambiguous"], br["m_corrected"]) == (None, None, None)
        assert m["admission"]["overclaim_rate"] is None

    def test_single_sample(self, store) -> None:
        """One decomposition, one child, declared ambiguous.

        D = 1, C = 1, A_d = 1, R = 0, E = 0, V = 1, A = 1.
        b_declared = 1.0, f_declared = 1.0, b_corrected = 1.0,
        f_ambiguous = 1.0, m_corrected = 1.0.
        """
        _decompose(store, "p0", declared=1, ambiguous=1)
        br = _metrics(store)["branching"]
        assert (br["b_declared"], br["f_declared"]) == (1.0, 1.0)
        assert (br["b_corrected"], br["f_ambiguous"], br["m_corrected"]) == (1.0, 1.0, 1.0)

    def test_all_ambiguous(self, store) -> None:
        """Every declared child is ambiguous: D = 2, C = 6, A_d = 6.

        b_declared = 3.0, f_declared = 1.0, V = 6, A = 6,
        b_corrected = 3.0, f_ambiguous = 1.0, m_corrected = 3.0 (badly supercritical).
        """
        _decompose(store, "p0", declared=4, ambiguous=4)
        _decompose(store, "p1", declared=2, ambiguous=2)
        br = _metrics(store)["branching"]
        assert br["b_declared"] == 3.0, br
        assert br["f_declared"] == 1.0, br
        assert br["b_corrected"] == 3.0, br
        assert br["f_ambiguous"] == 1.0, br
        assert br["m_corrected"] == 3.0, br

    def test_mixed_case_hand_computed(self, store) -> None:
        """The full mix, every value worked out by hand and pinned exactly.

        Decompositions (``decompose_outcome``)::

            node "c3": declared=4, ambiguous=1
            node "p0": declared=3, ambiguous=0
            node "p1": declared=3, ambiguous=2

            D   = 3
            C   = 4 + 3 + 3   = 10
            A_d = 1 + 0 + 2   = 3

        Admission dispositions (``admission_checked``, one per node)::

            c1  claimed=True   admitted
            c2  claimed=True   admitted
            c3  claimed=True   reclassify_decompose   -> R
            c4  claimed=False  escalate               -> E, E_ambiguous
            c5  claimed=True   escalate               -> E (overclaim, NOT ambiguous)
            c6  claimed=True   admitted
            c7  claimed=True   reclassify_decompose   -> R

            R = 2, E = 2, E_ambiguous = 1

        Corrected::

            V           = C - E                = 10 - 2 = 8
            A           = A_d + R - E_ambiguous = 3 + 2 - 1 = 4
            b_declared  = C / D = 10 / 3 = 3.3333333333333335
            f_declared  = A_d / C = 3 / 10 = 0.3
            b_corrected = V / D = 8 / 3  = 2.6666666666666665
            f_ambiguous = A / V = 4 / 8  = 0.5
            m_corrected = A / D = 4 / 3  = 1.3333333333333333

        Admission overclaim::

            claimed_atomic = c1,c2,c3,c5,c6,c7      = 6
            not admitted   = c3, c5, c7             = 3
            overclaim_rate = 3 / 6 = 0.5

        ``c3``'s reclassification produced a logged decomposition; ``c7``'s did
        not, so exactly one decomposition is unmeasured.
        """
        _decompose(store, "c3", declared=4, ambiguous=1)
        _decompose(store, "p0", declared=3, ambiguous=0)
        _decompose(store, "p1", declared=3, ambiguous=2)
        _admission(store, "c1", claimed=True, decision="admitted")
        _admission(store, "c2", claimed=True, decision="admitted")
        _admission(store, "c3", claimed=True, decision="reclassify_decompose")
        _admission(store, "c4", claimed=False, decision="escalate")
        _admission(store, "c5", claimed=True, decision="escalate")
        _admission(store, "c6", claimed=True, decision="admitted")
        _admission(store, "c7", claimed=True, decision="reclassify_decompose")

        m = _metrics(store)
        br = m["branching"]
        assert br["decompositions"] == 3, br
        assert br["decompositions_unmeasured"] == 1, br
        assert br["children_declared"] == 10, br
        assert br["children_ambiguous_declared"] == 3, br
        assert br["children_reclassified"] == 2, br
        assert br["children_escalated"] == 2, br
        assert br["children_viable"] == 8, br
        assert br["children_ambiguous_corrected"] == 4, br

        assert br["b_declared"] == 10 / 3, br
        assert br["f_declared"] == 0.3, br
        assert br["b_corrected"] == 8 / 3, br
        assert br["f_ambiguous"] == 0.5, br
        assert br["m_corrected"] == 4 / 3, br

        assert m["admission"]["claimed_atomic"] == 6, m["admission"]
        assert m["admission"]["rejected_or_reclassified"] == 3, m["admission"]
        assert m["admission"]["overclaim_rate"] == 0.5, m["admission"]
        assert m["admission"]["decisions"] == {
            "admitted": 3, "reclassify_decompose": 2, "escalate": 2,
        }, m["admission"]

    def test_m_is_exactly_expected_ambiguous_children_per_decomposition(self, store) -> None:
        """The gate's definition, asserted as an identity: m == b_corrected * f.

        Since b_corrected = V/D and f_ambiguous = A/V, their product collapses to
        A/D --- the expected number of ambiguous children per decomposition, which
        is what "m < 1 implies the recursion terminates" actually requires.
        """
        _decompose(store, "p0", declared=5, ambiguous=2)
        _decompose(store, "p1", declared=3, ambiguous=1)
        _admission(store, "c1", claimed=True, decision="reclassify_decompose")
        _admission(store, "c2", claimed=True, decision="escalate")

        br = _metrics(store)["branching"]
        # D=2, C=8, A_d=3, R=1, E=1, E_amb=0 -> V=7, A=4
        assert br["children_viable"] == 7, br
        assert br["children_ambiguous_corrected"] == 4, br
        assert br["b_corrected"] == 7 / 2, br
        assert br["f_ambiguous"] == 4 / 7, br
        assert br["m_corrected"] == 2.0, br
        assert br["m_corrected"] == pytest.approx(br["b_corrected"] * br["f_ambiguous"])


# --------------------------------------------------------------------------- D4


def _reference_percentile_bootstrap(values, *, statistic="mean", n_boot=1000,
                                    alpha=0.05, seed=0):
    """Independent textbook percentile bootstrap, written from the definition.

    Deliberately does NOT share code with :func:`sherpa.metrics.bootstrap_ci`.
    The median uses :func:`statistics.median` --- the actual median, which
    averages the two middle order statistics for even-sized samples.
    """
    rng = random.Random(seed)
    stats = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        if statistic == "mean":
            stats.append(math.fsum(sample) / n)
        else:
            stats.append(float(statistics.median(sample)))
    stats.sort()
    return stats[int((alpha / 2) * n_boot)], stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


SKEWED = [0.02, 0.03, 0.03, 0.05, 0.06, 0.08, 0.09, 0.11, 0.14, 0.19,
          0.22, 0.28, 0.35, 0.44, 0.55, 0.69, 0.86, 1.10, 1.60, 3.40]


class TestBootstrapCI:
    """D4: the previous test used ``[0.4] * 20``, where every resample is identical."""

    def test_constant_input_degenerates(self) -> None:
        """Boundary case kept, but it proves nothing on its own --- hence the rest."""
        assert bootstrap_ci([0.4] * 20) == (0.4, 0.4)

    def test_empty_input_returns_none(self) -> None:
        assert bootstrap_ci([]) is None

    def test_skewed_sample_brackets_the_mean_strictly(self) -> None:
        """A right-skewed sample: the 95% CI must straddle the point estimate.

        ``min``/``max`` of the raw data would give (0.02, 3.40); a correct
        percentile bootstrap of the MEAN gives a far tighter interval around
        mean = 0.5145.
        """
        mean = math.fsum(SKEWED) / len(SKEWED)
        assert mean == pytest.approx(0.5145)
        lo, hi = bootstrap_ci(SKEWED)
        assert lo < mean < hi, (lo, mean, hi)
        # Strictly inside the data range: this is what kills (min, max).
        assert min(SKEWED) < lo and hi < max(SKEWED), (lo, hi)
        # And genuinely narrow: the whole interval is well under the data spread.
        assert (hi - lo) < 0.5 * (max(SKEWED) - min(SKEWED)), (lo, hi)

    def test_matches_an_independent_percentile_bootstrap_exactly(self) -> None:
        """Same seed, same resampling scheme, same order statistics -> identical."""
        for stat in ("mean", "median"):
            got = bootstrap_ci(SKEWED, statistic=stat, n_boot=400, alpha=0.05, seed=7)
            want = _reference_percentile_bootstrap(SKEWED, statistic=stat, n_boot=400,
                                                   alpha=0.05, seed=7)
            assert got == want, (stat, got, want)

    def test_smaller_alpha_widens_the_interval(self) -> None:
        """alpha is the total tail mass: alpha=0.01 is a 99% CI, WIDER than 95%."""
        lo95, hi95 = bootstrap_ci(SKEWED, alpha=0.05, n_boot=2000, seed=3)
        lo99, hi99 = bootstrap_ci(SKEWED, alpha=0.01, n_boot=2000, seed=3)
        assert lo99 <= lo95 and hi99 >= hi95, ((lo99, hi99), (lo95, hi95))
        assert (hi99 - lo99) > (hi95 - lo95), ((lo99, hi99), (lo95, hi95))

    def test_seed_changes_the_result(self) -> None:
        a = bootstrap_ci(SKEWED, seed=1, n_boot=500)
        b = bootstrap_ci(SKEWED, seed=2, n_boot=500)
        assert a != b, (a, b)
        assert bootstrap_ci(SKEWED, seed=1, n_boot=500) == a  # deterministic per seed

    def test_median_differs_from_mean_on_a_skewed_sample(self) -> None:
        """Covers ``metrics.py``'s median arm, which had 0% coverage.

        median(SKEWED) = (0.19 + 0.22)/2 = 0.205 vs mean 0.5145, so the two
        intervals must not merely differ --- they must sit in different places.
        """
        assert statistics.median(SKEWED) == pytest.approx(0.205)
        mean_ci = bootstrap_ci(SKEWED, statistic="mean", n_boot=800, seed=5)
        med_ci = bootstrap_ci(SKEWED, statistic="median", n_boot=800, seed=5)
        assert mean_ci != med_ci, (mean_ci, med_ci)
        assert med_ci[1] < mean_ci[1], (med_ci, mean_ci)

    def test_median_uses_the_real_median_for_even_samples(self) -> None:
        """``sorted(s)[n//2]`` is the upper middle value, not the median.

        With the four-element sample below, every resample's median is an
        average of two order statistics, so a CI built from upper-middle values
        is systematically too high.  The lower endpoint is the tell: the true
        median of ``[0, 0, 10, 10]``-style resamples can be 0.0, but the upper
        middle value can never be below the second order statistic.
        """
        values = [0.0, 0.0, 10.0, 10.0]
        lo, hi = bootstrap_ci(values, statistic="median", n_boot=500, alpha=0.05, seed=11)
        ref_lo, ref_hi = _reference_percentile_bootstrap(
            values, statistic="median", n_boot=500, alpha=0.05, seed=11)
        assert (lo, hi) == (ref_lo, ref_hi), ((lo, hi), (ref_lo, ref_hi))
        assert lo == 0.0, lo  # attainable only with a true (averaging) median

    def test_unknown_statistic_is_loud(self) -> None:
        """Silently treating ``statistic='p90'`` as a median is a lie about the number."""
        with pytest.raises(ValueError, match="statistic"):
            bootstrap_ci(SKEWED, statistic="p90")

    def test_nominal_coverage_of_the_mean(self) -> None:
        """Deterministic coverage check: ~95% of 95% CIs must contain the true mean.

        200 independent samples of n=40 drawn from an exponential population with
        rate 1 (true mean 1.0), each given a 95% percentile-bootstrap CI.  Seeded
        end to end, so this is a fixed number, not a flaky one.  Percentile
        bootstrap under-covers a bit on skewed data; anything below ~0.85 would
        mean the interval construction is broken.
        """
        gen = random.Random(20250824)
        covered = 0
        trials = 200
        for t in range(trials):
            sample = [gen.expovariate(1.0) for _ in range(40)]
            lo, hi = bootstrap_ci(sample, n_boot=200, alpha=0.05, seed=t)
            covered += lo <= 1.0 <= hi
        rate = covered / trials
        assert 0.85 <= rate <= 1.0, rate


# --------------------------------------------------------------------------- D5


class TestUsageHonesty:
    """D5: "no model was consulted" must not render as "the model used 0 tokens"."""

    def test_unobserved_fields_are_none_not_zero(self, store) -> None:
        """The kernel's ``add_usage(attempts=1, nodes=1)`` never mentions tokens.

        So ``tokens``/``cost_usd`` were never measured and must read ``None``,
        while ``nodes``/``attempts`` are real measured totals.
        """
        _log(store, "usage_checkpoint", attempts=1.0, nodes=1.0)
        _log(store, "usage_checkpoint", attempts=1.0, nodes=2.0)
        usage = _metrics(store)["usage"]
        assert usage["attempts"] == 2.0, usage
        assert usage["nodes"] == 3.0, usage
        assert usage["tokens"] is None, usage
        assert usage["cost_usd"] is None, usage
        # The key set stays stable for downstream consumers.
        assert set(usage) == {"tokens", "cost_usd", "nodes", "attempts"}

    def test_measured_zero_is_reported_as_zero(self, store) -> None:
        """A model that really reported 0 tokens is NOT the same as no model."""
        _log(store, "usage_checkpoint", tokens=0.0, cost_usd=0.0)
        usage = _metrics(store)["usage"]
        assert usage["tokens"] == 0.0, usage
        assert usage["cost_usd"] == 0.0, usage
        assert usage["tokens"] is not None

    def test_real_token_totals_sum(self, store) -> None:
        _log(store, "usage_checkpoint", tokens=120.0, cost_usd=0.0012)
        _log(store, "usage_checkpoint", tokens=80.5, cost_usd=0.0008)
        usage = _metrics(store)["usage"]
        assert usage["tokens"] == 200.5, usage
        assert usage["cost_usd"] == pytest.approx(0.002), usage

    def test_aggregate_skips_unmeasured_tokens(self) -> None:
        """Two runs measured 100/300 tokens, one never measured any.

        total_tokens must be 400 over 2 measured runs --- not 400 over 3, and
        certainly not a median dragged toward 0 by a run that never reported.
        """
        reports = [
            {"terminal_status": "completed", "admission": {"overclaim_rate": 0.4},
             "branching": {"m_corrected": 0.8}, "usage": {"tokens": 100.0}},
            {"terminal_status": "failed", "admission": {"overclaim_rate": 0.6},
             "branching": {"m_corrected": 1.2}, "usage": {"tokens": 300.0}},
            {"terminal_status": "completed", "admission": {"overclaim_rate": None},
             "branching": {"m_corrected": None}, "usage": {"tokens": None}},
        ]
        agg = aggregate_run_reports(reports)
        assert agg["runs_aggregated"] == 3
        assert agg["total_tokens"] == 400.0, agg
        assert agg["runs_with_token_measurements"] == 2, agg
        assert agg["median_tokens_per_run"] == 300.0, agg  # sorted [100,300][2//2]
        # A run whose m could not be derived must not enter the gate as a 0.0.
        assert agg["m_values"] == [0.8, 1.2], agg
        assert agg["m_upper_bound_max"] == 1.2, agg

    def test_aggregate_with_no_measurements_at_all(self) -> None:
        agg = aggregate_run_reports([{"terminal_status": None, "admission": {},
                                      "branching": {"m_corrected": None},
                                      "usage": {"tokens": None}}])
        assert agg["total_tokens"] is None, agg
        assert agg["median_tokens_per_run"] is None, agg
        assert agg["runs_with_token_measurements"] == 0, agg
        assert agg["m_values"] == [], agg
        assert agg["m_upper_bound_max"] is None, agg

    def test_report_renders_unmeasured_as_na(self) -> None:
        """The markdown report must say "n/a", never print a fabricated 0."""
        suite = {
            "runs_aggregated": 1, "task_success_rate": None, "task_success_ci95": None,
            "mean_overclaim_rate": None, "overclaim_ci95": None, "total_tokens": None,
            "m_upper_bound_max": None,
        }
        runs = [{
            "run_id": "r1", "terminal_status": "completed",
            "admission": {"checked": 0, "overclaim_rate": None},
            "branching": {"m_corrected": None}, "usage": {"tokens": None},
        }]
        md = render_report_md(suite, runs)
        assert "| r1 | completed | 0 | n/a | n/a | n/a |" in md, md
        assert "- total tokens: n/a" in md, md
        assert "corrected m observed max: n/a" in md, md

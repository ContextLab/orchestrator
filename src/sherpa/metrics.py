"""Measurement projections for #492 §3/§6: corrected branching and overclaim.

Every metric here is DERIVED FROM THE EVENT LOG — never asserted, never taken
from a planner's self-description. #492 is explicit that "metrics are
projections of the event log" and that "an ``atomic`` step is an admitted
executable claim, not a planner label".

The preregistered viability gate is on the corrected reproduction number

    m = E[ambiguous children per decomposition]

so ``m < 1`` means the recursive decomposition terminates. Concretely, over a
run's log:

======================  ====================================================
symbol                  derivation
======================  ====================================================
``D``                   ``decompose_outcome`` events (one per decomposition)
``C``                   ``sum(children_declared)``
``A_d``                 ``sum(children_ambiguous)`` — the planner's own label
``R``                   nodes whose final admission decision was
                        ``reclassify_decompose`` while ``atomic_claimed``
``E``                   nodes whose final admission decision was ``escalate``
``E_a``                 those of ``E`` that were NOT ``atomic_claimed``
``V = C - E``           viable children: escalated proposals were refused, so
                        they are not children the recursion can descend into
``A = A_d + R - E_a``   corrected ambiguous children
======================  ====================================================

``b_declared = C/D`` and ``f_declared = A_d/C`` remain planner self-reports and
are labelled as such. The corrected figures incorporate admission:

    b_corrected = V / D          f_ambiguous = A / V          m = A / D

and ``m == b_corrected * f_ambiguous`` identically, which is what makes ``m``
the expected ambiguous fan-out per decomposition rather than an unrelated
product of two ratios.

Two honesty rules run through the whole module:

* A ratio with a zero denominator is ``None``, never ``0.0``. A metric nobody
  measured must not masquerade as a measured zero — that failure mode is
  exactly how 41 identically-zero ``m_values`` reached ``suite.json`` unnoticed.
* ``decompositions_unmeasured`` counts decompositions we KNOW happened (a node
  was reclassified for decomposition) but whose fan-out never reached the log.
  A nonzero value means the instrumentation is incomplete and ``b``/``f``/``m``
  are computed from a subset.

Kernel contract (the event fields consumed here)
------------------------------------------------
``admission_checked``   ``node_key``; ``payload.decision`` in
                        ``{"admitted", "reclassify_decompose", "escalate"}``;
                        ``payload.atomic_claimed`` (bool). One event per
                        check; the LAST event for a ``node_key`` is that
                        node's final disposition.
``decompose_outcome``   ``node_key`` (the parent that fanned out);
                        ``payload.children_declared`` (int);
                        ``payload.children_ambiguous`` (int). MUST be emitted
                        for every decomposition, including the reclassify path
                        where the parent node ends in state ``skipped``.
``usage_checkpoint``    ``payload`` keys among ``tokens``/``cost_usd``/
                        ``nodes``/``attempts``. A key that never appears in any
                        event is reported as ``None`` (unmeasured).
``run_terminal``        ``payload.status``.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

#: Fields ``usage_checkpoint`` may carry. Reported as ``None`` when no event
#: ever supplied them, so "no model was consulted" stays distinguishable from
#: "the model reported zero".
USAGE_FIELDS = ("tokens", "cost_usd", "nodes", "attempts")

_STATISTICS = ("mean", "median")


def run_metrics(events: list) -> dict[str, Any]:
    """Project one run's event log into the #492 measurement report."""
    admissions = [e for e in events if e.kind == "admission_checked"]
    claimed = [e for e in admissions if e.payload.get("atomic_claimed", False)]
    rejected_claims = [e for e in claimed if e.payload["decision"] != "admitted"]

    terminal = next((e for e in reversed(events) if e.kind == "run_terminal"), None)

    return {
        "run_id": None,
        "admission": {
            "checked": len(admissions),
            "claimed_atomic": len(claimed),
            "rejected_or_reclassified": len(rejected_claims),
            "overclaim_rate": _ratio(len(rejected_claims), len(claimed)),
            "decisions": _tally(e.payload["decision"] for e in admissions),
        },
        "branching": _branching(events, admissions),
        "terminal_status": terminal.payload.get("status") if terminal else None,
        "usage": _usage(events),
    }


def _branching(events: list, admissions: list) -> dict[str, Any]:
    """Corrected fan-out, derived from ``decompose_outcome`` AND admission.

    Admission verdicts are collapsed to one disposition per ``node_key`` (the
    last one wins) so that a node re-checked after an authority change is one
    child, not two.
    """
    disposition: dict[str, tuple[bool, str]] = {}
    for e in admissions:
        key = e.node_key if e.node_key is not None else f"<anon:{e.seq}>"
        disposition[key] = (bool(e.payload.get("atomic_claimed", False)),
                            str(e.payload["decision"]))

    reclassified = {k for k, (claim, dec) in disposition.items()
                    if claim and dec == "reclassify_decompose"}
    escalated = {k for k, (_claim, dec) in disposition.items() if dec == "escalate"}
    escalated_ambiguous = {k for k in escalated if not disposition[k][0]}

    measured: dict[str, tuple[int, int]] = {}
    for e in events:
        if e.kind != "decompose_outcome":
            continue
        key = e.node_key if e.node_key is not None else f"<anon:{e.seq}>"
        measured[key] = (int(e.payload.get("children_declared", 0)),
                         int(e.payload.get("children_ambiguous", 0)))

    n_dec = len(measured)
    children = sum(d for d, _a in measured.values())
    ambiguous_declared = sum(a for _d, a in measured.values())

    n_reclassified = len(reclassified)
    n_escalated = len(escalated)
    viable = children - n_escalated
    ambiguous_corrected = ambiguous_declared + n_reclassified - len(escalated_ambiguous)

    return {
        "decompositions": n_dec,
        # Decompositions we know happened (admission sent the node back) but
        # whose fan-out never reached the log. Nonzero => incomplete kernel
        # instrumentation, and the ratios below cover only a subset.
        "decompositions_unmeasured": len(reclassified - set(measured)),
        "children_declared": children,
        "children_ambiguous_declared": ambiguous_declared,
        "children_reclassified": n_reclassified,
        "children_escalated": n_escalated,
        "children_viable": viable,
        "children_ambiguous_corrected": ambiguous_corrected,
        # Planner self-reports, honestly labelled as such.
        "b_declared": _ratio(children, n_dec),
        "f_declared": _ratio(ambiguous_declared, children),
        # Post-admission measurements.
        "b_corrected": _ratio(viable, n_dec),
        "f_ambiguous": _ratio(ambiguous_corrected, viable),
        "m_corrected": _ratio(ambiguous_corrected, n_dec),
    }


def _usage(events: list) -> dict[str, float | None]:
    """Sum ``usage_checkpoint`` deltas, leaving never-reported fields ``None``."""
    totals: dict[str, float | None] = dict.fromkeys(USAGE_FIELDS, None)
    for e in events:
        if e.kind != "usage_checkpoint":
            continue
        for k, v in e.payload.items():
            if k in totals:
                totals[k] = (totals[k] or 0.0) + float(v)
    return totals


def _ratio(numerator: float, denominator: float) -> float | None:
    """``numerator/denominator``, or ``None`` when nothing was measured."""
    return (numerator / denominator) if denominator else None


def _tally(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def bootstrap_ci(
    values: list[float],
    *,
    statistic: str = "mean",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean or median of *values*.

    ``alpha`` is the total tail mass, so a SMALLER alpha yields a WIDER
    interval (alpha=0.01 is a 99% CI). Deterministic given *seed*.
    """
    if statistic not in _STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}; expected one of {_STATISTICS}")
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        if statistic == "mean":
            stats.append(math.fsum(sample) / n)
        else:
            # statistics.median, not sorted(sample)[n//2]: for even n the latter
            # is the upper middle order statistic, a different (biased) estimator.
            stats.append(float(statistics.median(sample)))
    stats.sort()
    lo_i = int((alpha / 2) * n_boot)
    hi_i = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return stats[lo_i], stats[hi_i]


def aggregate_run_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Suite-level roll-up. Unmeasured per-run values are skipped, not zeroed."""
    successes = [
        1.0 if r.get("terminal_status") == "completed" else 0.0
        for r in reports
        if r.get("terminal_status") is not None
    ]
    overclaims = [
        float(r["admission"]["overclaim_rate"])
        for r in reports
        if r.get("admission", {}).get("overclaim_rate") is not None
    ]
    tokens = [
        float(r["usage"]["tokens"])
        for r in reports
        if r.get("usage", {}).get("tokens") is not None
    ]
    ms = [
        float(r["branching"]["m_corrected"])
        for r in reports
        if r.get("branching", {}).get("m_corrected") is not None
    ]
    return {
        "runs_aggregated": len(reports),
        "task_success_rate": (math.fsum(successes) / len(successes)) if successes else None,
        "task_success_ci95": bootstrap_ci(successes),
        "mean_overclaim_rate": (math.fsum(overclaims) / len(overclaims)) if overclaims else None,
        "overclaim_ci95": bootstrap_ci(overclaims),
        # How many runs actually reported tokens, so a small total cannot be
        # mistaken for a cheap suite when it is really an uninstrumented one.
        "runs_with_token_measurements": len(tokens),
        "median_tokens_per_run": sorted(tokens)[len(tokens) // 2] if tokens else None,
        "total_tokens": math.fsum(tokens) if tokens else None,
        "m_values": ms,
        "m_upper_bound_max": max(ms) if ms else None,
    }


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _fmt(x: float | None, spec: str) -> str:
    return "n/a" if x is None else format(float(x), spec)


def render_report_md(suite: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    lines = ["# sherpa measurement report", "", "Raw projections from real runs; no assumed numbers.", ""]
    lines.append(f"- runs aggregated: {suite['runs_aggregated']}")

    sr = suite["task_success_rate"]
    ci = suite.get("task_success_ci95")
    sr_txt = _fmt_pct(sr) + ("" if ci is None else f" (CI95 {ci[0]:.2f}..{ci[1]:.2f})")
    lines.append(f"- task success rate: {sr_txt}")

    oc = suite.get("mean_overclaim_rate")
    oci = suite.get("overclaim_ci95")
    oc_txt = _fmt_pct(oc) + ("" if oci is None else f" (CI95 {oci[0]:.2f}..{oci[1]:.2f})")
    lines.append(f"- atomic overclaim rate: {oc_txt}")

    mb = suite.get("m_upper_bound_max")
    if mb is None:
        lines.append("- corrected m observed max: n/a")
    else:
        verdict = "subcritical (<1)" if mb < 1 else "SUPERCRITICAL (>=1)"
        lines.append(f"- corrected m observed max: {mb:.3f} — {verdict} on this fixture distribution")
    lines.append(f"- total tokens: {_fmt(suite['total_tokens'], '.0f')}")
    lines.append("")
    lines.append("| run | status | admissions | overclaim | m_corrected | tokens |")
    lines.append("|-|-|-|-|-|-|")
    for r in runs:
        adm = r["admission"]
        lines.append(
            "| {rid} | {st} | {n} | {ov} | {m} | {tok} |".format(
                rid=r.get("run_id", "-"),
                st=r.get("terminal_status"),
                n=adm["checked"],
                ov=_fmt_pct(adm["overclaim_rate"]),
                m=_fmt(r["branching"]["m_corrected"], ".3f"),
                tok=_fmt(r["usage"]["tokens"], ".0f"),
            )
        )
    lines.append("")
    return "\n".join(lines)

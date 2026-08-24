"""Measurement projections for #492 §3/§6: corrected branching and overclaim.

Every metric is derived from logged events — never asserted. ``m = b·f`` uses
the corrected fan-out (post-admission), per the issue's preregistered
definition: b is the mean number of viable children per decomposition and f
the fraction of children whose atomic claims did not survive admission.
"""

from __future__ import annotations

import math
import random
from typing import Any


def run_metrics(events: list) -> dict[str, Any]:
    admissions = [e for e in events if e.kind == "admission_checked"]
    claimed = [e for e in admissions if e.payload.get("atomic_claimed", False)]
    rejected_claims = [e for e in claimed if e.payload["decision"] != "admitted"]

    terminal = next((e for e in reversed(events) if e.kind == "run_terminal"), None)
    usage = {"tokens": 0.0, "cost_usd": 0.0, "nodes": 0, "attempts": 0}
    for e in events:
        if e.kind == "usage_checkpoint":
            for k, v in e.payload.items():
                if k in usage:
                    usage[k] += float(v)

    declared: list[int] = []
    ambiguous: list[int] = []
    for e in events:
        if e.kind == "decompose_outcome":
            declared.append(int(e.payload.get("children_declared", 0)))
            ambiguous.append(int(e.payload.get("children_ambiguous", 0)))

    n_dec = len(declared)
    total_children = sum(declared)
    b_declared = (total_children / n_dec) if n_dec else 0.0
    f_corrected = (sum(ambiguous) / total_children) if total_children else 0.0
    b_corrected = b_declared - b_declared * f_corrected
    m_corrected = b_corrected * f_corrected

    return {
        "run_id": None,
        "admission": {
            "checked": len(admissions),
            "claimed_atomic": len(claimed),
            "rejected_or_reclassified": len(rejected_claims),
            "overclaim_rate": (len(rejected_claims) / len(claimed)) if claimed else None,
            "decisions": _tally(e.payload["decision"] for e in admissions),
        },
        "branching": {
            "decompositions": n_dec,
            "b_declared": round(b_declared, 4),
            "f_ambiguous": round(f_corrected, 4),
            "b_corrected": round(b_corrected, 4),
            "m_corrected": round(m_corrected, 4),
        },
        "terminal_status": terminal.payload.get("status") if terminal else None,
        "usage": usage,
    }


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
    if not values:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        if statistic == "mean":
            stats.append(math.fsum(sample) / len(sample))
        else:
            stats.append(float(sorted(sample)[len(sample) // 2]))
    stats.sort()
    lo_i = int((alpha / 2) * n_boot)
    hi_i = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return stats[lo_i], stats[hi_i]


def aggregate_run_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
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
    tokens = [float(r.get("usage", {}).get("tokens", 0.0)) for r in reports]
    ms = [float(r["branching"]["m_corrected"]) for r in reports if "branching" in r]
    success_rate = (math.fsum(successes) / len(successes)) if successes else None
    overclaim_mean = (math.fsum(overclaims) / len(overclaims)) if overclaims else None
    return {
        "runs_aggregated": len(reports),
        "task_success_rate": success_rate,
        "task_success_ci95": bootstrap_ci(successes),
        "mean_overclaim_rate": overclaim_mean,
        "overclaim_ci95": bootstrap_ci(overclaims),
        "median_tokens_per_run": sorted(tokens)[len(tokens) // 2] if tokens else None,
        "total_tokens": math.fsum(tokens),
        "m_values": ms,
        "m_upper_bound_max": max(ms) if ms else None,
    }


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


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
    lines.append(f"- total tokens: {suite['total_tokens']:.0f}")
    lines.append("")
    lines.append("| run | status | admissions | overclaim | m_corrected | tokens |")
    lines.append("|-|-|-|-|-|-|")
    for r in runs:
        adm = r["admission"]
        lines.append(
            "| {rid} | {st} | {n} | {ov} | {m:.3f} | {tok:.0f} |".format(
                rid=r.get("run_id", "-"),
                st=r.get("terminal_status"),
                n=adm["checked"],
                ov=_fmt_pct(adm["overclaim_rate"]),
                m=float(r["branching"]["m_corrected"]),
                tok=float(r["usage"]["tokens"]),
            )
        )
    lines.append("")
    return "\n".join(lines)

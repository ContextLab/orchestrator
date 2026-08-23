#!/usr/bin/env python3
"""Measure `f` -- the parameter the whole #485 design rests on.

Recursive decomposition is a Galton-Watson branching process: a node emits `b`
steps, each ambiguous with probability `f`, and the process is finite in
expectation iff `m = b*f < 1`. Every review of #485 so far has had to GUESS
`f`. This script measures it, with a real planner, on real problems, against a
real capability library -- and measures how it moves as the library grows,
which is the claim that the library is the termination mechanism rather than an
efficiency nicety.

    .venv/bin/python scripts/prototypes/measure_ambiguity.py            # cached
    .venv/bin/python scripts/prototypes/measure_ambiguity.py --refresh  # re-call

Results are cached in measurements/ambiguity.json so the numbers are
reproducible without re-spending tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minikernel import LLMPlanner, load_env_key

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "measurements", "ambiguity.json")

# A capability pool that grows in realistic tiers. The point of the tiers is
# that `f` is NOT a property of the model: it is a property of
# (model, problem distribution, library contents).
TIERS: dict[str, list[str]] = {
    "L0_bare": [],
    "L1_minimal": ["read_file@1", "write_file@1", "run_python@1", "llm_transform@1"],
    "L2_working": [
        "read_file@1", "write_file@1", "run_python@1", "llm_transform@1",
        "web_search@1", "fetch_url@1", "extract_text@1", "summarize@1",
        "parse_csv@1", "sql_query@1", "regex_extract@1", "plot_chart@1",
    ],
    "L3_mature": [
        "read_file@1", "write_file@1", "run_python@1", "llm_transform@1",
        "web_search@1", "fetch_url@1", "extract_text@1", "summarize@1",
        "parse_csv@1", "sql_query@1", "regex_extract@1", "plot_chart@1",
        "ocr_pdf@1", "transcribe_audio@1", "translate@1", "embed_text@1",
        "vector_search@1", "dedupe_records@1", "join_tables@1",
        "statistical_test@1", "fit_model@1", "cross_validate@1",
        "git_clone@1", "run_tests@1", "diff_files@1", "render_markdown@1",
        "http_post@1", "schedule_job@1", "classify@1", "validate_schema@1",
    ],
}

PROBLEMS = [
    "Produce a literature review of recursive LLM task decomposition in which "
    "every citation resolves to a real paper and supports the claim it is "
    "attached to.",
    "Given a directory of YAML pipeline definitions, compile them to a typed "
    "intermediate representation and report which ones fail validation and why.",
    "Reproduce figure 3 of a published paper from the authors' released dataset "
    "and report where your numbers differ from theirs.",
    "Find the median household income of every US county and render a "
    "choropleth map of the result.",
    "Determine whether a proposed database schema change is safe to deploy "
    "against a 400M-row production table.",
    "Summarise the last 500 commits of a repository into a changelog grouped by "
    "user-visible behaviour change.",
    "Decide which of three candidate caching strategies to adopt for a service, "
    "with evidence.",
    "Extract every numeric claim from a 200-page PDF report and check each one "
    "against the underlying spreadsheet.",
    "Write and validate a regression test that reproduces an intermittent "
    "failure reported only in CI.",
    "Translate a technical manual from English to Japanese preserving all code "
    "blocks and cross-references verbatim.",
    "Estimate the annual carbon footprint of a company's cloud infrastructure "
    "from its billing exports.",
    "Given a corpus of 50,000 customer support tickets, identify the five "
    "product defects responsible for the most support load.",
    "Convert a legacy Fortran numerical routine to Python and prove the outputs "
    "agree to within floating-point tolerance.",
    "Design and run an experiment that determines whether a new ranking model "
    "improves user outcomes.",
    "Audit a codebase for hardcoded credentials and produce a remediation plan "
    "ordered by blast radius.",
    "Given a city's public transit GTFS feed, compute how many residents live "
    "within a 15-minute walk of frequent service.",
]


def measure(planner: LLMPlanner, tiers: dict[str, list[str]],
            problems: list[str]) -> list[dict]:
    rows = []
    for tier, caps in tiers.items():
        maturities = {c: "trusted" for c in caps}
        for i, problem in enumerate(problems):
            try:
                draft = planner.decompose(problem, "any->report", maturities, 0)
            except Exception as exc:
                print(f"    !! {tier} p{i}: {exc}")
                continue
            rows.append({
                "model": planner.model, "tier": tier, "n_caps": len(caps),
                "problem_index": i, "problem": problem[:90],
                "b": draft.fan_out, "ambiguous": draft.n_ambiguous,
                "f": round(draft.f, 4), "m": round(draft.fan_out * draft.f, 4),
                "tokens_in": draft.tokens_in, "tokens_out": draft.tokens_out,
                "rationale": draft.rationale[:160],
            })
            print(f"    {tier:12s} p{i:<2d} b={draft.fan_out} "
                  f"amb={draft.n_ambiguous} f={draft.f:.2f} "
                  f"m={draft.fan_out * draft.f:.2f}")
    return rows


def report(rows: list[dict]) -> None:
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["tier"], r["n_caps"])].append(r)
    print(f"\n{'='*84}")
    print("MEASURED decomposition statistics (real planner, real problems)")
    print(f"{'='*84}")
    # NOTE ON THE ESTIMATOR. The Galton-Watson offspring mean is
    # m = E[number of ambiguous children], i.e. mean(b*f) -- NOT mean(b) *
    # mean(f). The two differ whenever b and f are correlated across problems,
    # and here they are: the first version of this report used the product of
    # the means and called L2 "critical" when the correct estimator makes it
    # subcritical. Both are printed so the difference is visible.
    print(f"{'model':<16} {'library':<12} {'#caps':>5} {'mean b':>7} {'mean f':>7} "
          f"{'m (correct)':>12} {'mean-b*mean-f':>14} {'P(>=1 amb)':>11} {'regime':>14}")
    print("-" * 106)
    for (model, tier, n), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        b = statistics.mean(r["b"] for r in rs)
        f = statistics.mean(r["f"] for r in rs)
        m = statistics.mean(r["ambiguous"] for r in rs)
        naive = b * f
        pm = sum(1 for r in rs if r["ambiguous"] >= 1) / len(rs)
        regime = ("SUPERCRITICAL" if m > 1.05 else
                  "critical" if m > 0.95 else "subcritical")
        print(f"{model:<16} {tier:<12} {n:>5} {b:>7.2f} {f:>7.3f} {m:>12.2f} "
              f"{naive:>14.2f} {pm:>11.0%} {regime:>14}")
    print("-" * 84)
    hardest = sorted(rows, key=lambda r: -r["ambiguous"])[:5]
    print("\nhighest-m problems observed:")
    for r in hardest:
        print(f"  amb={r['ambiguous']}  b={r['b']} f={r['f']:.2f}  [{r['tier']}]  "
              f"{r['problem'][:70]}")
    easiest = [r for r in rows if r["ambiguous"] == 0]
    print(f"\n{len(easiest)}/{len(rows)} decompositions emitted NO ambiguous step "
          f"(m = 0, immediate termination)")
    tin = sum(r["tokens_in"] for r in rows)
    tout = sum(r["tokens_out"] for r in rows)
    print(f"total planner cost: {len(rows)} calls, {tin:,} in + {tout:,} out tokens")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--crosscheck", default="gpt-5.6-sol")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if os.path.exists(CACHE) and not args.refresh:
        rows = json.load(open(CACHE))
        print(f"(cached: {len(rows)} measurements from {CACHE})")
        report(rows)
        return 0

    key = load_env_key("OPENAI_API_KEY")
    if not key:
        print("No OPENAI_API_KEY reachable; cannot measure. "
              "This script refuses to invent numbers.")
        return 2
    rows: list[dict] = []
    print(f"measuring with {args.model} across {len(TIERS)} library tiers "
          f"x {len(PROBLEMS)} problems ...")
    rows += measure(LLMPlanner(args.model, key, args.base_url), TIERS, PROBLEMS)
    if args.crosscheck:
        print(f"\ncross-checking model dependence with {args.crosscheck} "
              f"at L2_working ...")
        rows += measure(LLMPlanner(args.crosscheck, key, args.base_url),
                        {"L2_working": TIERS["L2_working"]}, PROBLEMS)
    json.dump(rows, open(CACHE, "w"), indent=1)
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Design simulations for #485

Four standalone models of the architecture proposed in
[#485](https://github.com/ContextLab/orchestrator/issues/485), written to
support the design review posted at
[#485 (comment)](https://github.com/ContextLab/orchestrator/issues/485#issuecomment-5384176128).

**These are not product code.** Nothing under `src/` imports them and they are
not on any execution path. They exist so that the numbers quoted in that review
are reproducible, and so that a future agent can replace an assumption with a
measurement and see what moves.

They are also an instance of the thing #485 itself asks for in component 1c:
turn a proposed solution into deterministic operations over objects of the
assumed shape, write down the predicted observables in advance, run it, and see
which predictions survive.

## Running them

Standard library only — no orchestrator import, no network, no API keys.

```bash
.venv/bin/python scripts/simulations/decomposition_branching.py
.venv/bin/python scripts/simulations/scratchpad_contention.py
.venv/bin/python scripts/simulations/context_recursion.py
.venv/bin/python scripts/simulations/red_team_gate.py
```

Every stochastic model is seeded, so output is byte-identical run to run. The
invariants they demonstrate are asserted in
`tests/test_design_simulations.py` (marked `unit`), which is what keeps them
from rotting:

```bash
.venv/bin/python -m pytest tests/test_design_simulations.py -q
```

## What each one answers

| script | #485 component | review § | headline |
|-|-|-|-|
| `decomposition_branching.py` | 1 — recursive breakdown | §1, §5 | Decomposition is a Galton-Watson process: finite **iff `m = b·f < 1`**. At ~10 steps per pipeline, fewer than 1 step in 10 may be ambiguous, at every level. |
| `scratchpad_contention.py` | 3 — shared scratchpad | §2 | The LLM "consider" call sits inside the critical section, capping the whole fleet at `3600 / critical_section` steps/hour — 450/hr at 8s — regardless of fleet size. Re-summarising an append-only log per read is quadratic. |
| `context_recursion.py` | 1b — context recursion | §3 | The tree converges cheaply (depth ≤ 5 for 100M tokens, ~1.5× corpus cost), but retains `r**depth ≈ 0.0016` of the original at the top. It is a navigation structure, not a retrieval structure. |
| `red_team_gate.py` | 1a — critical review | §4 | The concern ledger removes scope drift entirely (8.6 → 3.6 rounds). But at `p_detect = 0.5`, only 45% of "clean" verdicts are truly clean. Same-family review has a hard `1-ρ` ceiling. |

### One finding that post-dates the review

Hardening `context_recursion.py` against its tests separated two claims the
first draft had collapsed together. The recursion **converges iff `r < 1`** —
that part of the review stands — but the *depth* that takes explodes as `r`
approaches 1: `r = 0.5` needs 5 levels, `r = 0.9` needs 29, `r = 0.99` needs
299. Only the first of those fits under any sane depth cap.

The useful corollary: top-level fidelity is `payload / N` **whatever `r` is**.
It is forced by the target size, not chosen. What `r` buys is *how many lossy
hops* you pass through to get there. So for a fixed final size, aggressive
single-hop summarisation accumulates less distortion than gentle multi-hop
summarisation — **prefer fewer, harder compressions**. That is a design
recommendation the review did not make, and it is asserted in
`test_top_level_fidelity_is_set_by_target_size_not_by_compression_ratio`.

## Assumptions, and how to replace them

Each script has an `ASSUMPTIONS` dict at module scope. **Every value in it is a
guess.** None was measured, because no live model was reachable when the review
was written: the `HF_TOKEN` in `~/.orchestrator/.env` authenticates against
`router.huggingface.co/v1/models` but returns `401` on `/v1/chat/completions`,
and there was no `DARTMOUTH_CHAT_API_KEY` on the machine.

| assumption | used by | how to measure it |
|-|-|-|
| `f` — P(a step is ambiguous) | `decomposition_branching` | **The most important number in the project.** Decompose ~50 real subproblems, count how many need recursion. Decides whether the architecture works at all. |
| `r` — summary compression per level | `context_recursion`, `scratchpad_contention` | Summarise a real corpus at a fixed prompt; take output/input tokens. |
| `critical_section_locked_s` | `scratchpad_contention` | Time one real read-tail + decide + write cycle. |
| `p_detect` — P(reviewer finds a given defect) | `red_team_gate` | Seed known defects into artifacts, count how many a reviewer returns. Easier under evidential gating: count submitted artifacts. |
| `p_regress` — P(a fix introduces a defect) | `red_team_gate` | Track defects introduced by fix commits. |
| `ρ` — family-wide blind-spot rate | `red_team_gate` | Give the same seeded artifact to two model families; measure the overlap of what each misses. |
| `red_team_rounds`, `red_team_tokens` | `decomposition_branching` | Fall out of `red_team_gate` once `p_detect` is measured. |

### Structural vs. numeric

The **structural** conclusions do not depend on any of the above, and should be
treated as load-bearing:

- termination iff `m = b·f < 1`;
- throughput ceiling of `1 / critical_section`, invariant to fleet size;
- quadratic-vs-linear summarisation churn on an append-only log;
- summary-tree fidelity decays as `r**depth`;
- catch rate for same-family reviewers is bounded above by `1 - ρ`.

The **numeric** results — leaf counts, dollar figures, round counts, percentages
— are only as good as the table above. Quote them with the assumption attached.

## If you change something

1. Edit the relevant `ASSUMPTIONS` entry (or add a row to a table in `main()`).
2. Re-run the script and `tests/test_design_simulations.py`.
3. If a *structural* claim moved, that is a finding — post it on #485 rather
   than quietly editing the number, so the design conversation sees it.

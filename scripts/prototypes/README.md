# `minikernel` — an executable schematic for #485

A deliberately small, runnable kernel for the redesign proposed in
[#485](https://github.com/ContextLab/orchestrator/issues/485). Not a product,
and not on the ADR-0001 path: nothing under `src/` imports it, and it imports
nothing from `src/`. It exists so the issue's claims can be **executed** rather
than argued about.

The three design reviews on #485 all reached their conclusions by modelling
mechanisms in isolation (see `scripts/simulations/`). This does the other half:
it puts every mechanism the issue proposes into one running system, in its
cheapest honest form, and reports what breaks.

```bash
.venv/bin/python scripts/prototypes/run_scenarios.py       # 55 end-to-end checks
.venv/bin/python -m pytest tests/test_minikernel.py -q     # 57 unit tests
.venv/bin/python scripts/prototypes/measure_ambiguity.py   # measured f (cached)
.venv/bin/python scripts/prototypes/probe_optimism.py      # is low f real? (cached)
```

## Modules

| module | #485 component | what it is |
|-|-|-|
| `store.py` | 3 | one substrate: append-only event log, content-addressed blobs, sealed-segment journal, summary DAG, FTS. The scratchpad, insight pool, context tables and tool history are **queries over it**, not four stores. |
| `ir.py` | 1 | typed plan IR (sequence / branch / bounded loop / call / decompose), its validator, `Authority`, and the `Budget` ledger. No `goto`. |
| `capabilities.py` | 3 | one lifecycle for tools, skills and reusable plans: `draft → candidate → trusted`, `quarantined`, `revoked`, plus the bug-report/triage workflow. |
| `library.py` | 1 + 3 | the solved-problem library — two-key retrieval (statement similarity **and** typed I/O signature) and, added after the harness demanded it, **negative results**. |
| `review.py` | 1a | separation of duty, frozen criteria, concern ledger, evidential gate, insight pool with contradiction detection. |
| `planner.py` | 1 | `StubPlanner` (deterministic, for sweeps) and `LLMPlanner` (a real model over stdlib `urllib`, for measurement). |
| `runtime.py` | 2 | the durable executor: nested runs, crash-resume, budget escalation, addressed message bus, admission control. |

## What the harness is for

Each scenario in `run_scenarios.py` asserts a property the design needs. When a
scenario failed, the **kernel** was changed, not the assertion — and six of
those changes are design findings, not typos:

1. **The library must store negative results.** A mission that ends at the depth
   cap taught the system nothing, so re-running it cost exactly as much,
   forever. The one regime where learning matters was the one regime where
   learning could not start. With dead-end memory a repeated unreachable
   mission costs 5 nodes instead of 20 — and still never reports success.
2. **An escalating sibling must not cancel the others.** Returning on the first
   unreachable subtree threw away every sibling that was still solvable, and
   with them everything the run would have learned.
3. **A wildcard in a stored signature defeats the two-key match.** A solution
   published as `any->any` matches every later query, so text similarity
   silently becomes the only key. Untyped solutions are no longer published.
4. **Budget exhaustion must not count against a cached plan.** It says nothing
   about whether the plan was right, but counting it as a failure dropped
   reliability below the retrieval floor after one unlucky mission.
5. **A planner must be a function of its inputs.** The first `StubPlanner`
   carried one RNG stream across calls, so asking the same question twice gave
   different plans and the library's benefit was unmeasurable.
6. **`atomic` has to be a checked claim, not a label** — see below.

## The measurement that matters

`f` — the fraction of steps a planner marks ambiguous — is the load-bearing
parameter: recursion is finite in expectation iff `m = b·f < 1`. Every prior
review had to guess it. Measured on 16 real problems across four library tiers
(`measurements/ambiguity.json`):

```
model            library      #caps  mean b  mean f  m (correct)   regime
gpt-5.4-mini     L0_bare          0    2.50   1.000         2.50   SUPERCRITICAL
gpt-5.4-mini     L1_minimal       4    2.73   0.483         1.33   SUPERCRITICAL
gpt-5.4-mini     L2_working      12    3.38   0.302         0.88   subcritical
gpt-5.4-mini     L3_mature       30    4.31   0.489         1.88   SUPERCRITICAL
gpt-5.6-sol      L2_working      12    4.80   0.013         0.07   subcritical
```

Two things to note. **The library helps and then hurts** — a bigger menu drives
`f` down but drives `b` up, and `m` is their product. And **the estimator
matters**: `m` is `E[ambiguous children]`, not `mean(b)·mean(f)`; the product of
the means called L2 critical when it is subcritical.

Then `probe_optimism.py` asks an independent judge model whether each
*declared-atomic* step can really be done by the capability it names:

```
planner        plans  mean b  declared m  overclaim rate  corrected m
gpt-5.4-mini       8    3.00        1.88  4/9  = 44%             2.38
gpt-5.6-sol        8    4.38        0.12  12/34 = 35%            1.62
```

The stronger planner's apparent `m` of 0.12 is really **1.62**. It was not less
ambiguous; it was more optimistic. Declared `f` is not a safety metric, and the
cheapest way for any planner to look like it terminates is to lie about what is
easy. That is why `Runtime` takes an `admission` hook, and why scenario S10
shows the same run reporting `completed` with a placeholder answer without it
and `escalated` with it.

## Assumptions

- Token counts are a `len/3.5` estimate, deliberately over-counting. A real
  implementation must use the selected model's tokeniser.
- `ReviewBoard.detection_rate_prior` (0.6) is a **guess**, carried through to
  every reported `residual_risk`. It should be replaced by a measured
  per-reviewer detection rate as soon as there is one.
- The measured `f` numbers are properties of *(model, problem distribution,
  library contents)* and of these 16 problems in particular. Quote all three.
- The judge in `probe_optimism.py` is one model (`gpt-5.5`). Its own error rate
  is unmeasured; the overclaim rates are therefore lower-confidence than the
  direction of the effect.

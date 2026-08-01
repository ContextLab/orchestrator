# ADR 0002: Retire the pipeline integration testing subsystem

**Status:** accepted
**Date:** 2026-08-01
**Closes:** #435
**Relates to:** #430 (retire the frozen competing implementations), #429
(consolidate the two parallel model-adapter layers), #354 (legacy suite),
#241 / #104 (`run` and `validate` disagree)

## Decision

Delete, rather than repair:

- `src/orchestrator/testing/pipeline_integration_infrastructure.py` (958 lines)
- `src/orchestrator/testing/pipeline_integration_demo.py` (544 lines)
- `tests/integration/test_pipeline_integration_infrastructure.py` (721 lines)

Replace only the intent with six tests against the canonical compiler and
orchestrator, in `tests/test_pipeline_model_contracts.py`.

## Why not repair it

The module could not construct its own central model. Its constructor passed
four separate keyword arguments that no core dataclass defines —
`ModelCapabilities(max_output_tokens=)`, `ModelRequirements(network_access=)`,
`ModelMetrics(tokens_per_second=)`, `ModelCost(input_cost_per_token=)`.
Correcting one revealed the next. It was written against an API that has never
existed, and therefore had never run: **27 tests, permanently red**.

Renaming those arguments was explicitly rejected. It would have made the first
layer pass while preserving a second testing architecture — its own model,
provider, validator, scoring system and orchestration facade — inside the
shipped package, with no user-facing consumer. That is the same
parallel-implementation problem #430 exists to remove, and repairing it would
have grown the surface that work has to retire.

Supporting facts, all verified rather than assumed:

- **No external consumers.** `rg` for the module names and for every symbol
  they export (`PipelineTestModel`, `PipelineTestProvider`,
  `PipelineIntegrationResult`, …) returns nothing outside the three files
  themselves. `src/orchestrator/testing/__init__.py` does not import them.
- **It duplicated working infrastructure.** `tests/test_infrastructure.py`
  already provides a deterministic model and provider.
- **It shipped pretend models.** It hardcoded fictional `openai/gpt-4` and
  `anthropic/claude-*` entries into a package that end users install.
- **It contradicted project policy.** It was built around `mock_responses`,
  while CONTRIBUTING states "No mock objects. Ever."

Git history preserves the code. An archive copy inside the tree would add
noise without reducing risk.

## What replaced it

Six tests, hermetic, using the existing deterministic model:

| Test | Pins |
|-|-|
| `test_a_valid_pipeline_compiles` | the canonical compiler accepts a well-formed pipeline |
| `test_a_malformed_pipeline_is_refused` | validation failure raises, not compiles-to-broken |
| `test_a_model_pipeline_executes` | a model step actually runs |
| `test_structured_output_pipeline_returns_an_object` | structured output is a mapping |
| `test_execution_failure_is_reported_not_swallowed` | a failing step reports failure |
| `test_a_model_pipeline_compiles` | **xfail(strict)** — documents #241 |

Test-only models and providers stay under `tests/`. They are not product code
and must not be shipped again.

## Defects this uncovered

Deleting the parallel architecture forced the replacement tests through the
canonical path, which immediately exposed three real bugs that the retired
subsystem had been routing around:

1. **`select_model()` results were looked up again.** Five call sites in
   `hybrid_control_system.py` and `model_based_control_system.py` did
   `model = get_model(await select_model(...))`. `select_model` already
   returns a `Model`, so `get_model()`'s `":" in model_name` check raised
   `TypeError: argument of type 'Model' is not a container`. Every
   registry-selected `action: generate` step failed.
2. **`create_test_orchestrator()` built the wrong registry.** Two classes
   share the name `ModelRegistry`; only `models.registry` has
   `register_provider`, and only `models.model_registry` has
   `can_provide_models`, which `Orchestrator.__init__` calls. The helper used
   the first, so it raised on every call and could not construct an
   orchestrator at all. (The duplicate-registry split itself is #429.)
3. **The deterministic model advertised the wrong task names.** It offered
   `text-generation`, while the control system asks for `generate` and real
   providers advertise `generate` — so it was ineligible for every generate
   step and selection failed with `NoEligibleModelsError`.

None of these were visible while a second testing stack stood in for the real
one. That is the strongest argument for the deletion: the parallel
architecture was not merely unused, it was hiding the state of the supported
path.

## Consequences

- 2,223 lines removed; 6 tests added, in the **blocking** layer rather than a
  permanently red one.
- 27 tests were **retired, not fixed**. #354's backlog shrinks by that amount
  for that reason, and must not be read as progress on the legacy suite.
- `validate` still rejects model pipelines that `run` accepts. That divergence
  is now pinned by a strict xfail instead of being absorbed by a bespoke
  validator, and closing #241 will make that test flip to a visible failure,
  prompting its removal.

## If systematic example-pipeline validation is wanted again

Build it around the canonical `orchestrator validate` path so it returns the
same diagnostics as the CLI. It must not introduce another model, provider,
validator, scoring system, or orchestration facade — that is what was just
removed.

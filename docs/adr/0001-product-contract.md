# ADR 0001: Product and Architecture Contract

- **Status:** Accepted
- **Date:** 2026-07-30
- **Supersedes:** the completion claims in `CLAUDE_SKILLS_REFACTOR_COMPLETE.md`,
  `IMPLEMENTATION_STATUS.md`, and the multi-provider promises in the current `README.md`
- **Context:** `notes/2026-07-30_repository_recovery_audit_and_plan.md`

## Why this exists

The repository accumulated several overlapping architectures and generated
completion narratives faster than it validated them. There was no authoritative
answer to "which code path is the product?", so every subsystem looked equally
load-bearing and nothing could be safely deleted.

This ADR fixes one contract. Anything not named here is not part of the
supported product, regardless of what other documents in the tree claim.

## The supported user journey

Exactly one:

> A user writes a YAML pipeline, runs it through the CLI or the Python API, and
> gets a typed result plus a trace of what executed.

```text
YAML source
    │
    ▼  YAMLCompiler.compile()          canonical compiler
Pipeline (immutable spec: Task graph + metadata)
    │
    ▼  Orchestrator.execute_pipeline() canonical executor
Typed result + outputs + state + trace
```

## Canonical implementations

These are the only implementations the product supports. Where competitors
exist, they are listed so they can be retired against contract tests.

| Role | Canonical | Status of alternatives |
|-|-|-|
| Compiler | `compiler/yaml_compiler.py::YAMLCompiler` | `compiler/enhanced_yaml_compiler.py`, `graph_generation/` — **off the import closure**, safe to retire |
| Control-flow compiler | `compiler/control_flow_compiler.py` | — |
| Executor | `orchestrator.py::Orchestrator` | `engine/`, `execution/`, `api/` are off the closure. **`executor/` and `runtime/` are NOT** — `orchestrator.py` imports `executor.parallel_executor` and `runtime`, so they are dependencies of the canonical path, not competitors |
| Domain model | `core/pipeline.py`, `core/task.py` | — |
| Model registry | `models/model_registry.py` | `models/registry.py` also loads on **every** import (`models/__init__.py`), so two classes named `ModelRegistry` coexist. Not yet separable |
| Tool registry | `tools/base.py::ToolRegistry` | `tools/registry.py`, `tools/universal_registry.py` — off the closure |
| State | `state/state_manager.py` | `state/langgraph_state_manager.py` is **imported** at module scope by `orchestrator.py`; only its *instantiation* is opt-in |
| Expression evaluation | `core/expressions.py` (AST-based, fail-closed) | Migrated: `control_flow/auto_resolver.py` (the canonical condition path), `enhanced_condition_evaluator`, `runtime/dependency_resolver`, `actions/condition_evaluator`, `engine/advanced_executor`, `auto_resolution/integration`, `tools/pipeline_recursion_tools`, `execution/engine`. **Still on `eval()`: `control_systems/hybrid_control_system.py`** (its transform expressions need `json.loads`, comprehensions and generator expressions) |

Alternatives that are genuinely off the import closure are **frozen**: no new
features, no new callers, removed once contract tests characterize the behavior
worth keeping.

Rows marked as still-coupled are a statement of fact, not intent: decoupling
them is remaining work, and this table must not claim otherwise. An earlier
version of this table listed `executor/`, `runtime/`, `models/registry.py` and
the LangGraph state manager as merely "competing", which an audit disproved by
tracing the actual import closure.

Selection rule: never choose an implementation by its name. `enhanced`,
`advanced`, `clean`, `original`, `backup` and `v2` carry no information. Run
candidates through the same contract tests and keep the smallest one that
passes.

## Provider policy

- Core interfaces stay **provider-neutral**.
- Two providers are being brought under live acceptance tests:
  - **Anthropic**, matching the October refactor and the code that works.
  - **Dartmouth Chat**, an OpenAI-compatible gateway that serves several
    models at zero cost per token. It was added because unfunded real-model
    coverage is worth more than mocked coverage, and it needs no provider
    extra: the adapter speaks the gateway's HTTP API with `aiohttp`, already
    a core dependency. Free/paid status is read from the live catalog, and
    paid models are refused unless `ORCHESTRATOR_ALLOW_PAID_MODELS=1`.
- A provider earns the word **supported** only when its `live-tests` job
  passes remotely. As of 2026-08-01:
  - **Dartmouth Chat: supported.** `live-dartmouth` passed with 9 tests
    against real free models. This is the first provider to clear the bar.
  - **Anthropic: not supported.** Its live job is red — the account has no
    credit, so the API returns 400 and the tests correctly refuse to pass
    rather than reporting unverified behaviour as working. See #432.
- OpenAI, Google, HuggingFace and Ollama adapters remain in the tree but are
  **unsupported** until they have contract tests and live acceptance tests.
- No provider may be advertised in the README as supported before its live
  job passes. Describing it as verified-locally-but-not-gated is permitted,
  provided the README says exactly that.

## Dependency policy

A package belongs in `dependencies` only if it is required to import
`orchestrator` and run a pipeline that uses deterministic local tools.

Core (12): pydantic, pyyaml, jinja2, jsonschema, networkx, click,
python-dotenv, aiofiles, aiohttp, requests, psutil, numpy.

Everything else is an extra: `anthropic`, `openai`, `google`, `langgraph`,
`web`, `multimedia`, `viz`, `infra`, `crypto`.

**Import rule:** no optional dependency may be imported at module scope in any
module reachable from `import orchestrator`. Optional deps are imported inside
the function that needs them and raise an `ImportError` naming the extra. A
missing extra must degrade one feature, never break the package import.

## Public surface

- Python: `orchestrator.Orchestrator`, `orchestrator.YAMLCompiler`,
  `orchestrator.Pipeline`, `orchestrator.Task`, `orchestrator.compile`
- CLI: `orchestrator run <pipeline.yaml>` and `orchestrator validate <pipeline.yaml>`
- The CLI and the Python API must produce identical results for identical input.

Exit codes: `0` success, `1` execution failure, `2` validation/compile failure,
`130` interrupted.

## The result contract

`Orchestrator.execute_pipeline` returns a `PipelineResult`
(`core/pipeline_result.py`). It is a `Mapping`, so `result["step_id"]` returns
the step's raw value exactly as before; the trace arrives as attributes
alongside.

| | |
|-|-|
| `status` / `success` | whether the run as a whole succeeded |
| `outputs` | declared `outputs:`, resolved |
| `steps` | `StepResult` per step |
| `execution_order` / `execution_levels` | dependency order, and what could run together |
| `started_at` / `completed_at` / `duration` | run timing |
| `failed_steps` / `skipped_steps` / `retried_steps` | the trace, by category |

Each `StepResult` carries its canonical action, status, success, value,
structured error (`error` and `error_type`), the tool or model and provider
that ran it, start/end/duration, retry count and dependencies.

**`status` and `success` are not the same question.** `status` records whether
the task finished; `success` whether it worked. A tool that returns
`{"success": False}` without raising *finishes* — reading only the status
reported a failing pipeline as successful, and that is why the CLI's exit code
consults `success`.

Declared outputs do not change the shape of the return value. They used to:
a pipeline with `outputs:` returned `{"steps": …, "outputs": …}` and one
without returned `{step_id: …}`, so a caller could not index a result without
first checking which it had been handed, and a step named `outputs` collided
with the second.

`to_dict()` is the stable serialisation the CLI emits — no `default=str`
coercion, so it round-trips. `normalized()` drops the execution id and
wall-clock times, which are the only fields that legitimately differ between
two runs. **The CLI and the Python API must produce equal normalised
documents**, compared whole rather than by selected nested values.

## Actions and template resolution

A step names either a tool and an operation on it (`tool: filesystem` with
`action: read`), or an action the runtime executes itself (`action: generate`).
`core/actions.py` is the single source of truth for the second group. Each
action is an `ActionSpec` carrying its canonical name, aliases, handler,
whether it needs a model or a tool, its required parameters and its result
schema. Everything else is *derived* from that registry rather than restated
beside it:

| Consumer | Derived as |
|-|-|
| Executor dispatch | `resolve_action(...).handler` |
| Validator recognition | `is_known_action(...)` |
| Advertised `supported_actions` | `SUPPORTED_ACTIONS` |
| Alias normalisation | `canonical_action(...)`, applied by the compiler |
| Documentation | `docs/actions.md`, generated and drift-tested |

So `validate` and `run` cannot disagree about whether an action exists (#241),
and the vocabulary cannot drift apart again.

**An unrecognised action is refused.** It used to become a prompt for the
model, so `action: gernate` returned a plausible answer and reported success.
It now fails at compile time, and again at dispatch — the runtime checks
independently, because a caller can construct a `Task` and reach execution
without going through YAML validation at all. `<AUTO>...</AUTO>` remains
supported: that is an author explicitly asking the model to interpret an
instruction, which is not the same as a typo falling through.

Aliases are accepted but deprecated. The compiler rewrites them to the
canonical name and warns, so exactly one spelling reaches the task graph and
the trace.

Template rendering deliberately falls back to returning the original text when
a reference is undefined, because resolution runs in several passes and a
reference that cannot resolve yet may resolve later. That fallback must not
survive the handoff to a tool:

> A template reference that reaches a tool still unresolved fails the step
> before any side effect. No file is written, the step's envelope carries
> `success: false` naming the reference, and the run exits 1.

The test is *survival*, not presence: a marker in the rendered output that was
not in the input is content, not a failure. Jinja's own escape `{{ '{{' }}`
renders to a literal `{{`, and a step result may legitimately contain text that
looks like a template (#153).

Rendering is all-or-nothing per string. One undefined reference aborts the
whole render, so every marker in that string is reported, not just the bad one.

## Test layers

| Layer | Marker | Network | Secrets | Runs in default CI |
|-|-|-|-|-|
| Unit | `unit` | no | no | yes |
| Contract | `contract` | no | no | yes |
| Integration | `integration` | local services | no | on demand |
| Live provider | `live` | yes | yes | scheduled/manual only |
| End-to-end | `e2e` | no | no | yes (against the built wheel) |

Deterministic fakes are **permitted and expected** at the unit and contract
layers. The earlier "no mocks, real models only" policy conflated deterministic
unit testing with live acceptance testing; it made the suite slow, costly and
irreproducible, and it is withdrawn. Live-provider tests remain mandatory for
claiming provider support — they just are not the default layer.

Tests import `orchestrator`, never `src.orchestrator`. Loading the package under
two identities produces duplicate singleton registries and divergent class
identities.

No test may mutate the host machine. Installing or starting Docker during test
collection is prohibited.

## Golden pipelines

Four executable acceptance specifications. The first two are hermetic; the last
two are `live` and skip without a credential:

1. **`basic`** — deterministic local tools, sequential steps, template
   interpolation, typed outputs.
2. **`control-flow`** — parallel fan-out and a dependent fan-in join, with
   template interpolation across step results, plus a deliberately failing step
   to verify failure propagation and exit codes. Conditional branching and
   loops are *not* covered by this fixture and are not yet part of the
   supported contract; see #333 (`on_false` / `on_success`) and #320.
3. **`live-anthropic`** — the same shape as `basic` but with one real Anthropic
   call. Marked `live`, skipped unless `ANTHROPIC_API_KEY` is set.
4. **`live-dartmouth`** — the same shape, against a free Dartmouth Chat model.
   Marked `live`, skipped unless `DARTMOUTH_CHAT_API_KEY` is set. Its job runs
   separately from the Anthropic one: run together, a missing Dartmouth
   credential produced skips inside a green Anthropic job, which read as
   coverage that did not exist.

Golden pipelines run through **both** the CLI and the Python API and must agree.

## Scope: what is deferred

Not part of the first recovery release; frozen, not deleted:
dashboards and monitoring, self-healing maintenance, multimedia breadth,
marketplaces, automatic skill creation, POML, RouteLLM, Deep Agents,
production deployment tooling, `web/`, `analytics/`, `admin/`.

## Open decisions

- **Issue #218 (rename the toolbox):** explicitly **deferred**. The
  distribution stays `py-orc` and the import package stays `orchestrator` for
  the first alpha. Revisiting this before a verified baseline exists would
  churn packaging and docs for no validation gain.

## Status claims

Support and readiness claims must be generated from recorded CI evidence, not
prose. Until a tagged artifact passes install, security, integration and
quickstart gates, the project describes itself as **alpha**. The
`Development Status :: 3 - Alpha` classifier is correct; README text claiming
production readiness is not.

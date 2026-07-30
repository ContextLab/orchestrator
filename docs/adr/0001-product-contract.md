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
- **Anthropic** is the first and only supported live provider, matching the
  October refactor and the code that actually works.
- OpenAI, Google, HuggingFace and Ollama adapters remain in the tree but are
  **unsupported** until they have contract tests and live acceptance tests.
- No provider may be advertised in the README before those tests pass.

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

Three executable acceptance specifications, all hermetic:

1. **`basic`** — deterministic local tools, sequential steps, template
   interpolation, typed outputs.
2. **`control-flow`** — conditional branching and parallel fan-out with
   dependency ordering, plus a deliberately failing step to verify failure
   propagation and exit codes.
3. **`live-anthropic`** — the same shape as `basic` but with one real Anthropic
   call. Marked `live`, skipped unless `ANTHROPIC_API_KEY` is set.

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

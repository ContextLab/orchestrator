# sherpa — user guide

sherpa is the experimental, evidence-driven recursive agentic runtime from
[issue #492]. It lives beside the frozen supported `orchestrator/` path;
nothing under `src/orchestrator/` depends on it. Design decisions, identities,
event semantics, and explicit non-goals are recorded in
[ADR 0003](adr/0003-sherpa-recursive-agentic-runtime-mvp.md). Package-level
reference: [src/sherpa/README.md](../src/sherpa/README.md).

## Install / layout

The package ships inside the `py-orc` source tree (`pip install -e .` picks it
up; a `sherpa` console script is registered alongside `orchestrator`). Core
dependencies: stdlib + pydantic — no provider extras required for hermetic use.

## Quickstart

```python
from pathlib import Path
from sherpa import Engine, ProblemSpec

spec = ProblemSpec(
    id="demo",
    goal="append an audited line",
    authority={"fs_read": ["**"], "fs_write": ["**"]},
    metadata={"root_nodes": [
        {"kind": "invoke_capability", "id": "w", "capability": "fs.write_file",
         "inputs": {"path": "out.txt", "content": "hello"}},
        {"kind": "return", "id": "fin", "outputs": {"ok": True}},
    ]},
)

engine = Engine(Path("./ws"))
result = engine.run(spec)          # RunResult(status="completed", outputs={...})
```

Every side effect is bracketed by events in one append-only SQLite log
(`ws/sherpa.db`, WAL) with content-addressed blobs in `ws/blobs/`. Kill the
process at any point; `engine.resume(run_id)` rebuilds by replay, skips nodes
already completed, and finishes without repeating effects.

## CLI

```bash
python -m sherpa run problem.json --workspace ws     # 0 completed / 1 loud failure
python -m sherpa resume <run_id> --workspace ws
python -m sherpa status <run_id> --workspace ws
python -m sherpa export-trace <run_id> trace.json --workspace ws
```

Problem specs are JSON serializations of `sherpa.ir.ProblemSpec`; plans are
authored by planners (deterministic `StubPlanner` over a declarative library,
or `LLMPlanner` over a model channel) and validated against the IR schema plus
parent authority before execution.

## Models

No API keys are needed: hermetic runs replay model responses through
`RecordedChannel`, and `EchoChannel` fails loudly if a run unexpectedly needs a
model. With credentials present, `channel_policy="live"` uses orchestrator's
supported providers (Dartmouth Chat free models, then HuggingFace Inference
API) — retired providers are never contacted.

## Acceptance suite and benchmarks

```bash
.venv/bin/python -m pytest tests/sherpa -q    # requires Python >= 3.11 (repo requirement)
.venv/bin/python -m pytest tests/sherpa/test_acceptance.py # three demonstrations
.venv/bin/python -m sherpa.benchmarks.harness --out benchmarks/artifacts   # full matrix
```

The harness writes raw run artifacts plus `report.md` and evaluates the
preregistered go/no-go gates from issue #492. Thresholds are MVP decisions on
the declared fixture distribution, not product claims; failing gates retain
negative evidence by design.

[issue #492]: https://github.com/ContextLab/orchestrator/issues/492

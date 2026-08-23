# sherpa — experimental recursive agentic runtime (issue #492)

sherpa is a new, experimental package that lives beside the frozen supported
`orchestrator/` path. It answers issue #492's question — *can a small
event-sourced kernel plus checked recursive planning solve bounded, externally
verifiable tasks reliably enough to justify productionizing?* — with measured
executions, not transcripts.

Design decisions and non-goals: [ADR 0003](../docs/adr/0003-sherpa-recursive-agentic-runtime-mvp.md).

## One-command acceptance suite

```bash
.venv/bin/python -m pytest tests/sherpa -q
```

Hermetic: no network, no API keys. Real SQLite (WAL), real filesystem blobs,
real SIGKILL fault injection, real pytest subprocesses. The model boundary is
the one recorded surface (`RecordedChannel`), per the issue's replay rule for
external boundaries.

## Python API

```python
from pathlib import Path
from sherpa import Engine, ProblemSpec

engine = Engine(Path("./workspace"))          # + channel_policy="live" for live models
result = engine.run(problem_spec)             # -> RunResult(status=..., outputs=...)
resumed = engine.resume(result.run_id)        # crash/pause/attended-answer recovery
engine.export_trace(run_id, Path("trace.json"))
```

## CLI

```bash
python -m sherpa run problem.json --workspace ws
python -m sherpa resume <run_id> --workspace ws
python -m sherpa status <run_id> --workspace ws
python -m sherpa export-trace <run_id> trace.json --workspace ws
```

Exit codes mirror orchestrator: 0 completed, 1 any loud non-completed terminal.

## Benchmarks and measurement report

```bash
.venv/bin/python -m sherpa.benchmarks.harness --out benchmarks/artifacts
```

Runs the three preregistered demonstrations plus a decomposition battery,
writes raw artifacts (`scenario_a/b/c.json`, per-run traces, `suite.json`) and
`report.md`, then evaluates the go/no-go gates from the issue:

| gate | threshold |
|-|-|
| decomposition decisions with independent admission outcomes | ≥ 50 |
| claimed-atomic steps admitted/rejected independently | ≥ 30 |
| held-out repair/corpus tasks externally verified within budgets | ≥ 80% |
| corrected branching m = b·f upper bound on fixture distribution | < 1.0 |
| seeded-needle retrieval recall | ≥ 95% |
| crash/resume preserves projections; no repeated effects | required |
| repair results verified by REAL pytest outside the runtime | no false successes |

Thresholds are preregistered MVP decisions on this fixture distribution, not
product claims. Failing gates retain negative evidence by design.

## Module map

| module | role |
|-|-|
| `ir.py` | typed plan IR: nine structural node variants, budgets, authority |
| `expr.py` | fail-closed AST-whitelist expression evaluator (no `eval`) |
| `events.py` | closed event vocabulary for the append-only log |
| `store.py` | SQLite WAL store + content-addressed blobs + FTS5 + leases |
| `capabilities.py` | typed capabilities: authority, executable probes, built-ins |
| `admission.py` | atomic-admission control (existence → I/O → authority → probe) |
| `planner.py` | deterministic plan signatures; Stub/LLM planners |
| `context.py` | journal, exact-span chunking, cited summary DAG, retrieval |
| `review.py` | bounded independent review; evidenced blocking findings only |
| `metrics.py` | corrected branching, overclaim rate, bootstrap CIs, reports |
| `kernel.py` | durable Engine: run / resume / status / export_trace |
| `benchmarks/` | scenarios A/B/C fixtures + measurement harness |

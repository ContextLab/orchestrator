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

`run_capability` is the only invocation path. It validates the resolved inputs
against the capability's declared `input_schema` **and** the returned value
against its `output_schema`; a capability that returns something it did not
declare raises `CapabilityContractError` rather than propagating the value.
Capability inputs are redacted before they are written to the event log, so a
secret passed as an input does not end up in `ws/sherpa.db`.

Crash injection is opt-in per `Engine`: the SIGKILL hook fires only when the
engine was constructed as `Engine(ws, fault_injection=True)`. Setting
`SHERPA_KILL_AFTER_EVENTS` in the environment on its own does nothing.

## Authority: what a grant actually covers

`Authority` has four dimensions — `fs_read`, `fs_write`, `net_domains`,
`subprocess_allow` — and `sherpa.authority` is the single implementation that
answers every question about them. Filesystem grants use this syntax:

| grant | covers |
|-|-|
| `src` or `src/` | `src` and everything beneath it (literal prefix subtree) |
| `src/*` | exactly one segment beneath `src` (`src/a.txt`, not `src/pkg/b.txt`) |
| `src/**` | any depth beneath `src`, including `src` itself |
| `src/a.txt` | that one file |
| `**` | everything beneath the **workspace** — not the whole disk |
| `/**` | the entire filesystem; the only way to ask for it |

Relative grants are anchored at the run's workspace, so the quickstart's
`{"fs_read": ["**"], "fs_write": ["**"]}` is *workspace-relative*: it cannot
reach `/etc/passwd`. Reaching outside the workspace has to be spelled out, as
`/**` or as an explicit absolute subtree such as `/srv/data/**`. Paths are fully
resolved (`..`, absolute paths, symlinks) *before* the grant comparison, so
`src/../../etc/passwd` under a grant of `src` is denied. Empty grant lists deny
everything; there is no implicit default.

Two different questions are kept apart:

- **Delegation** (`authority_covers`) asks whether a child plan may hold a
  pattern set at all, given the parent's. A child of a `src/**` parent may hold
  `src/pkg/**`; it may not hold `**`. `kernel._author_child` refuses an authored
  child plan that fails this check.
- **Access** (`path_within_grants`) asks whether one fully-resolved path may be
  touched. This runs per invocation, inside `run_capability`.

`CapabilitySpec.requires` names authority *dimensions*, not patterns. It is a
coarse gate: the dimension must be granted *something*. The per-resource check
happens per invocation against the resolved path, so a scoped grant such as
`fs_write=["out/**"]` is usable (previously any scoped grant made every fs
capability unusable). One caveat, measured: the built-in fs probes write a
uniquely-named canary at the **workspace root** during admission, so a grant
that excludes the workspace root — `["out/**"]` alone — fails the probe and the
step is reclassified for decomposition rather than run. Grant the root as well
(e.g. `["out/**", "*"]`) when scoping to a subdirectory.

Authority containment is enforced in `kernel` (delegation), `capabilities`
(per-invocation, in `run_capability`), and `admission` (before a leaf runs).
`ir.validate_plan` is structural only — it checks ids, branch/loop shape,
fan-out caps, estimated depth, and capability existence; it does **not** check
authority.

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

## Component walkthrough (real output)

`src/sherpa/demos.py` executes every subsystem against real fixtures and prints
what actually happened:

```bash
PYTHONPATH=src .venv/bin/python -m sherpa.demos > docs/examples/demo-walkthrough.txt
```

The committed transcript lives at
[docs/examples/demo-walkthrough.txt](examples/demo-walkthrough.txt).

**Re-running is not byte-reproducible, and the transcript says why.** Two lines
change on every run and nothing else does (verified by diffing two consecutive
runs): line 5, the workspace path, which is a fresh `tempfile.mkdtemp()`; and
the `resumed run run_...` line, whose run id is random per run. Everything else
— the plan listing, the expression verdicts, the event sequence and its seq
numbers, `events=43`, every admission decision, the terminal statuses, the whole
metrics block, the SIGKILL exit code, and the effect lists before and after
resume — is stable across runs.

The excerpt below is copied byte-for-byte out of the committed transcript
(source lines 12-15, 20-21, 26, 89, 93, 95, and 138-143), including its spacing:

```text
  step 1: [invoke_capability ] scan          fs.list_dir  (claimed-atomic)
  step 2: [invoke_capability ] read_entry    fs.read_file  (claimed-atomic)
  step 3: [invoke_capability ] write_report  fs.write_file  (claimed-atomic)
  step 4: [return            ] fin           {"audited": true}  (terminal)

evaluate('files_scanned >= threshold', {...}) -> True
injection attempt rejected: ExpressionError: disallowed syntax: Call

[repo-audit] terminal=completed events=43 replay_matches_live=True

     57 admission_checked            root_overclaim.sneaky_write escalate
     61 run_terminal                                  escalated
terminal status = 'escalated'  (loud failure, exit code would be 1)

victim process exit code = -9 (-9 => killed by SIGKILL)
effects on disk at death : ['effect-1']

resumed run run_1b6b11083949: terminal=completed replay_matches_live=True
effects after resume     : ['effect-1', 'effect-2', 'effect-3']
exactly-once             : True (no effect repeated despite the hard kill)
```

What each section proves:

| section | component | demonstrated behavior |
|-|-|-|
| 1 | `ir` + planner contract | goal -> typed plan; every side effect is a *claimed*-atomic step |
| 2 | `expr` | whitelist evaluator computes comparisons; call syntax (injection) is rejected before execution |
| 3 | kernel + store + capabilities | append-only events, admission pipeline (existence -> I/O schema -> authority -> executable probe), projection == replay-by-projection |
| 4 | admission authority | a step whose capability requires more authority than granted escalates LOUDLY - never silently skipped |
| 5 | metrics | branching/overclaim/usage derived purely from logged events |
| 6 | kernel durability | forked victim dies by real SIGKILL mid-run (`exit code = -9`) with `effect-1` already on disk; `engine.resume()` replays the log, finishes the remaining steps, and the effect file reads `['effect-1', 'effect-2', 'effect-3']` - each written exactly once |

Crash/resume under SIGKILL is additionally exercised across randomized kill
points by `tests/sherpa/test_kernel.py`.

## Limitations

These are measured properties of the MVP as it stands, not aspirations. Every
number below comes from the regenerated `benchmarks/artifacts/suite.json`.

**The branching gate is not yet exercised.** All 41 `m_values` in `suite.json`
are `0.0`, and `m_upper_bound_max` is `0.0`. That is a genuine measurement, not
a placeholder: the metric is derived from `admission_checked` events, and a
ratio with a zero denominator is reported as `None` rather than `0.0`, so these
zeros mean the numerator really was zero. But the fixture distribution contains
**no ambiguous children** (`children_ambiguous_declared` and
`children_ambiguous_corrected` are 0 throughout), so `m = b·f` is bounded
trivially. The gate does not yet test the branching it is meant to bound.

**No tokens are measured.** `total_tokens` is `null` and
`runs_with_token_measurements` is `0`, because hermetic runs consult no model —
they replay through `RecordedChannel`. Any cost or token claim about sherpa is
currently unmeasured.

**The recursion exercised is shallow.** Across all 41 benchmark run databases
the maximum persisted node `depth` is **1** — one level of child plan beneath
the root. The `demos.py` walkthrough is shallower still: every node is at depth
0, because its plans are given as explicit `root_nodes` and never decompose.
Deep recursion is supported by the kernel (depth is tracked, `max_depth` is
enforced, `AskUser` below the root escalates) but is not what the fixtures
measure.

**Issue #485 Component II (organization tree) is only partially realized.**
Parent/child linkage and depth are persisted per node (`nodes.parent_key`,
`nodes.depth`) and are rebuilt by `Store.replay_projection`. What does not
exist: any per-node agent identity — `nodes.owner_session` records a leased
worker session, and one session may execute many nodes while a node may see
several sessions across attempts — and any message *routing*. Delivery is
exact-key: `enqueue_message(run_id, to_node_key, ...)` and
`take_messages(run_id, node_key)` match `to_node_key` for equality, so the
caller must already know the recipient's node key. There is no addressing,
forwarding, broadcast, or backpressure (those are #487).

**Issue #485 Component III is not implemented.** There is no shared scratchpad
of agent thinking, no red-teamed insights pool, and no shared tool pool. The
operational journal in `context.py` is deliberately *not* a chain-of-thought
scratchpad: it stores concise typed entries (`intent`, `decision`,
`observation`, `assumption`, `blocker`, `result`), and required private
reasoning is explicitly out of scope.

**The solution cache stores no negative results.** ADR 0003 records "the
solution cache must store negative results" as a design finding, and
`Store.cache_put` accepts negative entries and refuses to record a
budget-exhausted run as evidence against a plan. But the kernel only ever writes
`{"status_class": "solved", ...}` (`kernel.py`, at the close of a successful
decomposition). Nothing in the runtime caches a failure today, so learning
cannot start in the regime the finding is about. Known gap.

**Replay rebuilds the run, not the corpus.** `Store.replay_projection` rebuilds
run status/error, `parent_run_id`, every node (state, `owner_session`, `depth`,
`parent_key`), usage totals, pending message count, and the run's finding ids.
It does **not** rebuild the document substrate: `chunks_meta`/`chunks_fts`,
`summaries`, or `solution_cache`, even though `chunk_indexed` and
`summary_created` events are logged. Those tables are written directly and are
not reconstructible from the log alone.

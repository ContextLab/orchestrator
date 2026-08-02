# Supported examples

Every pipeline in this directory is **tested as product behaviour**. Each one:

1. compiles under `orchestrator validate`;
2. executes through the CLI;
3. executes through the Python API;
4. produces the same normalised result document from both surfaces;
5. produces the artifacts and declared outputs its expectations record.

`tests/test_supported_examples.py` enforces all five, and fails if a file is
added here without an entry declaring what it should do — so the set cannot
grow past its coverage.

| Example | Demonstrates |
|-|-|
| `01_hello_filesystem.yaml` | typed parameters, templating, dependencies, declared outputs |
| `02_parallel_fanout_fanin.yaml` | independent steps sharing an execution level, joined by a dependent step |
| `04_conditions.yaml` | `condition:`, `on_false` routing, skipped steps |
| `06_failure_policy.yaml` | `timeout`, bounded retry, `on_failure: continue`, honest exit code |
| `07_templates_and_outputs.yaml` | step-result references and declared outputs |

Everything else under `examples/` is **not** in this set. Those pipelines
predate the current contract and most of them do not compile — 108 of 111 at
the time of writing (#104). Treat them as historical until each is proven and
moved here.

## Running one

```bash
orchestrator validate examples/supported/01_hello_filesystem.yaml
orchestrator run      examples/supported/01_hello_filesystem.yaml
```

`06_failure_policy.yaml` exits **1** on purpose: it contains a step that
fails, and the run reports that rather than hiding it.

## Not here yet

Numbering follows the planned set, so gaps are deliberate rather than lost:

- `03_data_etl` — needs the data-processing tool under the same contract.
- `05_loop` — `for_each`/`while` exist but are not yet pinned by contract tests.
- `08`–`09`, `11` — model pipelines. They cannot run hermetically in the
  blocking layer; they belong in the live job.
- `10_model_fallback` — **deliberately absent.** There is no model fallback:
  `select_model` raises rather than substituting a model the pipeline did not
  ask for. An example must not be written for behaviour that does not exist.
- `12`–`14` — checkpoint/resume, sub-pipelines and web research, each of which
  needs its own acceptance job first.

# Supported examples

Every pipeline in this directory is **tested as product behaviour**. Each one:

1. compiles under `orchestrator validate`;
2. executes through the CLI;
3. executes through the Python API;
4. produces the same normalised result document from both surfaces —
   *whole documents*, including every step value and the declared outputs,
   with only wall-clock times and the execution id blanked;
5. produces exactly the artifacts its case records, **with their exact
   contents**, and no others;
6. records exactly which steps completed, which were skipped and which
   failed, on every branch it has.

`tests/test_supported_examples.py` enforces all six, and fails if a file is
added here without a case declaring what it should do — so the set cannot grow
past its coverage.

A pipeline with a branch declares one case per branch, so the arm a default
run does not take is still executed in CI.

| Example | Demonstrates |
|-|-|
| `01_hello_filesystem.yaml` | typed parameters, templating, dependencies, declared outputs |
| `02_parallel_fanout_fanin.yaml` | independent steps sharing an execution level, joined by a dependent step |
| `04_conditions.yaml` | `condition:`, `on_false`/`on_success` routing, two **exclusive** branches converging on a join |
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

`04_conditions.yaml` takes its long branch by default. Force the other one
with a shorter input, and note that the two branches produce different files:

```bash
orchestrator run examples/supported/04_conditions.yaml               # long.txt
orchestrator run examples/supported/04_conditions.yaml -i content=hi # short.txt
```

Every run also writes a checkpoint under `./checkpoints/` in the working
directory. That is a side effect of running any pipeline rather than of these
examples.

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

# Tutorials

These tutorials teach the behavior Orchestrator currently verifies. Each lesson uses a pipeline in [`examples/supported/`](../../examples/supported/README.md), where tests compile and run it through both the CLI and Python API.

## Learning path

1. [Build and run a pipeline](getting-started.md)
2. [Compose parallel work](parallel-work.md)
3. [Route with conditions](conditional-routing.md)
4. [Use templates and declared outputs](templates-and-outputs.md)
5. [Recognize a reported failure](reported-failure.md)
6. [Handle timeouts and failures](failure-handling.md)

The first four tutorials succeed with exit code `0`. The final two deliberately exit `1` so that you can inspect both forms of honest failure.

## What happened to the old catalog?

The previous catalog generated 43 pages from legacy files under `examples/`. Most of those examples predate the current product contract and are not tested as supported behavior. Their generated pages remain in the source tree for historical comparison, but they are not tutorials and are not included in this learning path. An example returns here only after it has a contract test and moves into `examples/supported/`.

## Tutorial standard

Every lesson names its prerequisites, exact command, expected result, side effects, and next experiment. See the [documentation writing style](../writing-style.md) for the review checklist.

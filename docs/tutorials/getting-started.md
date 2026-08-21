# Build and run a pipeline

In this tutorial, you will validate and run a two-step pipeline that writes a greeting to a file and reads it back. The example uses only the base package.

## Prerequisites

- Python 3.11 or newer
- Orchestrator installed with `pip install py-orc`, or an editable checkout
- A shell opened at the repository root

## Read the pipeline

Open [`examples/supported/01_hello_filesystem.yaml`](../../examples/supported/01_hello_filesystem.yaml). The pipeline declares a `greeting` parameter, writes its value with the `filesystem` tool, then reads the file in a dependent step.

The dependency matters: `read_back` cannot start until `write_greeting` finishes. In other words, list order alone does not define execution order.

## Validate before running

```bash
orchestrator validate examples/supported/01_hello_filesystem.yaml
```

Validation compiles the YAML and checks its task graph without running a tool. A valid pipeline exits `0`; a compile or validation error exits `2`.

## Run with an input

```bash
orchestrator run examples/supported/01_hello_filesystem.yaml -i greeting=hi
cat output/greeting.txt
```

The command exits `0`, and `output/greeting.txt` contains `hi, world`. Orchestrator also writes a checkpoint below `checkpoints/`.

## Run through Python

```python
import asyncio

from orchestrator import Orchestrator
from orchestrator.control_systems.tool_integrated_control_system import ToolIntegratedControlSystem


async def main():
    runner = Orchestrator(control_system=ToolIntegratedControlSystem())
    try:
        result = await runner.execute_yaml_file(
            "examples/supported/01_hello_filesystem.yaml",
            {"greeting": "hi"},
        )
        print(result.outputs)
    finally:
        await runner.shutdown()


asyncio.run(main())
```

The CLI and Python API return the same normalized result. The Python result is a `PipelineResult`: it behaves like a mapping for step values and also exposes outputs, steps, timing, and execution-order metadata.

## Next

Continue with [Compose parallel work](parallel-work.md) to see how dependencies create execution levels.

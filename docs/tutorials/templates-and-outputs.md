# Use templates and declared outputs

This tutorial follows values between steps and exposes selected values as pipeline outputs. It uses [`07_templates_and_outputs.yaml`](../../examples/supported/07_templates_and_outputs.yaml).

## Run the example

```bash
orchestrator validate examples/supported/07_templates_and_outputs.yaml
orchestrator run examples/supported/07_templates_and_outputs.yaml
```

Both commands exit `0`. The second step refers to the first step's result with a template expression. The pipeline's `outputs` block then selects values for callers without changing the shape of the result object.

An unresolved template is allowed during intermediate rendering passes, because its value may not exist yet. If it reaches a tool unresolved, the step fails before the tool can perform a side effect.

## Next

Continue with [Recognize a reported failure](reported-failure.md).

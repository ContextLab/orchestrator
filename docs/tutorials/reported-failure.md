# Recognize a reported failure

This tutorial examines a tool that reports failure without raising an exception. It uses [`05_reported_failure.yaml`](../../examples/supported/05_reported_failure.yaml).

## Run the example

```bash
orchestrator validate examples/supported/05_reported_failure.yaml
orchestrator run examples/supported/05_reported_failure.yaml
```

Validation exits `0`; execution exits `1`. The `read_missing` step asks the filesystem tool to read a file that does not exist. The tool returns a failed result instead of raising, so the task status is `completed` while its step-level `success` value is `false`.

The distinction matters. Status answers whether the task finished. Success answers whether its work succeeded. The pipeline result and CLI exit code use the latter, so the failure is not hidden. Because the step says `on_failure: continue`, the `after` step still runs.

## Inspect the trace

Compare `result.steps["read_missing"].status` with `result.steps["read_missing"].success`. Then inspect `result.failed_steps`: a non-raising reported failure has a different trace shape from the raised timeout in the next tutorial.

## Next

Continue with [Handle timeouts and failures](failure-handling.md).

# Handle timeouts and failures

This tutorial inspects an intentional timeout without hiding it. It uses [`06_failure_policy.yaml`](../../examples/supported/06_failure_policy.yaml).

## Run the example

```bash
orchestrator validate examples/supported/06_failure_policy.yaml
orchestrator run examples/supported/06_failure_policy.yaml
```

Validation exits `0`. Execution exits `1` by design: one step exceeds its timeout, and `on_failure: continue` lets later work run without converting the failed step into a success.

The timed-out command is killed and reaped. The result records the failed step, its timeout, and any retry count.

## Understand the retry limit

`max_retries` currently bounds total attempts, despite its name. For example, `max_retries: 2` means one initial attempt and one retry. Values `0` and `1` both result in one attempt.

## Choose a failure policy

- `fail` aborts after a raised failure.
- `continue` runs later steps but preserves the failed result and exit code.
- `skip` and `retry` select their corresponding policies.
- A non-reserved `on_failure` value names a forward routing target.

A known gap remains: a tool can return `{"success": false}` without raising, and fail-fast scheduling does not yet stop immediately for that case. The final result and CLI exit code still report failure.

## Next

Review the [supported examples contract](../../examples/supported/README.md), then use the [YAML configuration guide](../user_guide/yaml_configuration.rst) as a reference while composing your own pipeline.

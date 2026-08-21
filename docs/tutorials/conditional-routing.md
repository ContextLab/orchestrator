# Route with conditions

This tutorial runs both branches of a conditional pipeline. It uses [`04_conditions.yaml`](../../examples/supported/04_conditions.yaml).

## Run the default branch

```bash
orchestrator run examples/supported/04_conditions.yaml
```

The default input takes the long branch and writes `long.txt`. The short-branch step is recorded as skipped; it is not treated as a failure.

## Run the other branch

```bash
orchestrator run examples/supported/04_conditions.yaml -i content=hi
```

This run writes `short.txt` and skips the long branch. Both runs exit `0`.

The source step uses `on_false` and `on_success` to jump forward. Steps between the source and target are skipped, and the branches converge on a join. Routing targets must exist and must point forward; the compiler rejects invalid or self-referential targets.

## Next

Continue with [Use templates and declared outputs](templates-and-outputs.md).


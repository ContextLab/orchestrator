# Compose parallel work

This tutorial shows how dependencies express parallelism and synchronization. It uses [`02_parallel_fanout_fanin.yaml`](../../examples/supported/02_parallel_fanout_fanin.yaml).

## Run the example

```bash
orchestrator validate examples/supported/02_parallel_fanout_fanin.yaml
orchestrator run examples/supported/02_parallel_fanout_fanin.yaml
```

Both commands exit `0`. Two independent write steps occupy the same execution level. A later join step depends on both, so it starts only after both writes finish.

## Inspect the result

The serialized result includes `execution_levels`. Use that field to inspect what the scheduler was allowed to run together; do not infer concurrency from the order in which log messages happen to appear.

For example, add a third independent write step and add its id to the join step's dependencies. Validation should place all three writes in one level and the join in the next.

## Next

Continue with [Route with conditions](conditional-routing.md).


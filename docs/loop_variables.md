# Loop Variables

Each loop construct binds its own names, in its own places. There is no single
list that applies to all of them, and the same name can be bound in a step's
body while being unavailable in the expression that starts the loop.

The tables here are the ones in `src/orchestrator/core/loop_contracts.py`, and
`tests/test_loop_runtime_parity.py` checks them by *executing* a pipeline for
every name: a name listed here renders in a real run, and a name absent from a
construct's table does not.

## `for_each`

Iterates a collection.

```yaml
steps:
  - id: process_items
    for_each: "{{ items }}"
    steps:
      - id: save
        tool: filesystem
        action: write
        parameters:
          path: "output/{{ item }}_{{ index }}.txt"
          content: |
            Processing {{ item }} at position {{ position }}
            This is item {{ index }} of {{ length }}
            First: {{ is_first }}  Last: {{ is_last }}
```

Bound in the body:

|name|meaning|
|-|-|
|`item`|the current item|
|`index`|zero-based iteration number|
|`is_first`, `is_last`|whether this is the first or last item|
|`position`|one-based position (`index + 1`)|
|`length`|number of items|
|`remaining`|items after this one|
|`has_next`, `has_prev`|whether a next or previous item exists|
|`loop_id`|the loop's identifier|
|`$loop_name`|the loop's name — this one has no bare spelling|

**Not bound in the iterable.** `for_each: "{{ item.children }}"` cannot work:
the collection has to be evaluated before there is an item to bind. Validation
rejects it.

## `while`

Repeats until a condition goes false.

```yaml
steps:
  - id: retry
    while: "{{ iteration < 3 }}"
    max_iterations: 10
    steps:
      - id: attempt
        parameters:
          note: "attempt {{ position }} of loop {{ loop_id }}"
```

Bound in the body: `iteration`, `index`, `is_first`, `position`, `loop_id`,
`loop_name`, `loop_state`.

Bound in the `while:` and `until:` conditions: **`iteration` and `loop_state`
only**. Unlike a `for_each` iterable, a condition is re-evaluated every
iteration, so it can see the counter — but it sees only what the loop handler
puts in scope at that moment, which is less than the body gets.

A `while` loop walks no collection, so it binds no `item`, `length` or
`is_last`.

## `action_loop`

Repeats a list of actions rather than walking a collection.

```yaml
steps:
  - id: poll
    action_loop:
      - action: filesystem
        parameters:
          action: write
          path: "out/{{ iteration }}.txt"
          content: "attempt {{ iteration }}"
    until: "{{ iteration >= 3 }}"
    max_iterations: 5
```

Bound in the body: `iteration`, `is_first`, `loop_id`, `has_previous`,
`total_duration`, `termination_reason`. No `item`, `index` or `position` —
there is no collection and no position in one.

The `action_loop` key holds the body, so those names are available inside it.

> **Known defect:** the `until:` condition is required but never evaluated, so
> the loop always runs a single iteration. See issue #476.

## `create_parallel_queue`

Generates a queue and runs actions across it in parallel.

```yaml
steps:
  - id: fan_out
    action: create_parallel_queue
    create_parallel_queue:
      "on": "{{ work_items }}"
      action_loop:
        - action: filesystem
          parameters:
            action: write
            path: "out/{{ index }}.txt"
            content: "{{ item }} of {{ queue_size }}"
```

Bound in the actions: `item`, `index`, `is_first`, `is_last`, `queue`,
`queue_size`, `parallel_queue_id`, `parent_task`.

**Not bound in `on:`** — that expression generates the queue, so nothing
per-item exists while it runs.

Two things about this construct differ from steps elsewhere: `on` must be
quoted (YAML 1.1 reads a bare `on` as the boolean `true`), and the nested
actions use `action:` rather than `tool:`.

## The `$` spelling

`{{ $item }}` works. `{{ $position }}` is a compile error:

```
unexpected char '$' at 3
```

`$` is not Jinja syntax. It works for some names only because a preprocessing
step rewrites them before rendering, and that rewrite covers a fixed list:
`$item`, `$index`, `$is_first`, `$is_last`, `$iteration`, `$loop_id`,
`$loop_name`, `$loop_state`.

**Prefer the bare spelling.** It is what the runtime actually resolves and what
every table above is written in. The one exception is `$loop_name` in a
`for_each` body, where the bare form is not bound. The inconsistency is issue
#474.

## Nested loops

An inner loop sees its own bindings and the enclosing loop's, including in the
inner loop's own iterable:

```yaml
steps:
  - id: outer
    for_each: "{{ categories }}"
    loop_name: outer_loop
    steps:
      - id: inner
        for_each: "{{ item.entries }}"   # the outer loop's item
        steps:
          - id: process
            parameters:
              category: "{{ $outer_loop.item }}"
              entry: "{{ item }}"        # the inner loop's item
```

The key for naming a loop is `loop_name:`. A named loop's variables are
reached as `{{ $<name>.<variable> }}`.

> **Known defect:** that spelling runs — the pipeline above writes `A` and `B`
> — but `orchestrator validate` rejects it with `unexpected char '$' at 3`,
> because validation parses the raw text while the runtime rewrites `$` first.
> See issue #474.

## One loop per step

A step declares one loop construct. A step carrying two — `for_each` and
`while` together, say — is rejected: which one would win is decided by
declaration order inside the validator, and no engine agrees to that order.

## `foreach` is not `for_each`

`foreach:` is recognised by the declarative engine's spec objects but is not
expanded by the control-flow compiler, so a `foreach` step either fails schema
validation or runs its body exactly once with nothing bound. Use `for_each`.
See issue #475.

## Troubleshooting

**A template appears unrendered in the output.** The name is not bound where
you used it. Check the construct's table above, and check whether you are in a
source expression (`for_each:`, `create_parallel_queue.on`) rather than a body.

**`index` starts at 0.** Use `position` for one-based numbering.

**`{{ $something }}` fails to compile.** Use the bare spelling; see above.

# eval() → constrained evaluator migration

Repo: `/Users/jmanning/orchestrator`, branch `recovery/phase-0-4`.
Target API: `orchestrator.core.expressions.{evaluate_condition, evaluate_expression, ExpressionError}`.

**I committed nothing.** Read "Concurrent-agent overlap" at the bottom before acting on
this report: another agent was working the same task in the same working tree at the
same time. It committed my in-flight edits inside its own commits, and it independently
migrated two of the sites I had reported as unmigratable by *extending*
`core/expressions.py`. The final-state table below describes the tree as it stands now;
the "mine" column says which changes I made.

## Final state: 8 of 9 sites migrated, 1 live `eval()` left

| # | Site | Status | Mine? |
|-|-|-|-|
| 1 | `tools/pipeline_recursion_tools.py:497` | MIGRATED | No — I migrated, hit a real failure, reverted, reported UNMIGRATED; the other agent then added `SAFE_METHODS` to the evaluator and migrated it |
| 2 | `runtime/dependency_resolver.py:281` | MIGRATED | Yes |
| 3 | `actions/condition_evaluator.py:118` (`BooleanEvaluator`) | MIGRATED | Yes |
| 4 | `actions/condition_evaluator.py:414` (`ExpressionEvaluator`) | MIGRATED | No — I reported UNMIGRATED (needs `**`); the other agent added bounded `ast.Pow` and migrated it |
| 5 | `control_flow/auto_resolver.py:747` | MIGRATED | Yes |
| 6 | `control_flow/enhanced_condition_evaluator.py:297` | MIGRATED | Yes |
| 7 | `control_systems/hybrid_control_system.py:722` | **UNMIGRATED** | Mine — documented in place, see below |
| 8 | `engine/advanced_executor.py:120` | MIGRATED | Yes — **most severe fix in the set** |
| 9 | `auto_resolution/integration.py:265` | MIGRATED | Yes — **was FAIL-OPEN, now fail-closed** |

`grep -n "\beval("` over `src/orchestrator` now returns exactly one live call site:
`control_systems/hybrid_control_system.py:728`. Remaining textual hits are a docstring
in `core/expressions.py` and blocklist string literals in `security/langchain_sandbox.py`
and `control_flow/condition_models.py`.

## Security-relevant behavior changes (from my six sites)

### Site 9 — was FAIL-OPEN, now fail-closed. Users can notice this.

`AutoTagIntegration._to_boolean` previously ended:

```python
try:
    return bool(eval(lower, {"__builtins__": {}}, {}))
except:
    return bool(value)          # every non-empty string → True
```

Any AUTO-resolved answer that was not a recognizable boolean expression — "maybe",
"probably not", any prose sentence a model returns — fell through to `bool(value)` and
became **True**, running the branch it was supposed to gate. Now:

```python
return evaluate_condition(lower, {}, default=False)
```

*User-visible change:* an `<AUTO>` tag whose resolved text is not a boolean or a boolean
expression now yields `False` instead of `True`, so conditions gated on such an answer
stop firing. Values that already evaluated (`"3"`, `"1 > 0"`, `"[1,2]"`) are unchanged.

### Site 8 — the worst RCE surface in the set

`ConditionalExecutor._evaluate_expression` called `bool(eval(expression))` with **no
globals argument at all**, so the expression saw the full `builtins` module *and*
`advanced_executor`'s module globals. `__import__('os').system(...)` in a pipeline
condition executed. Now `evaluate_condition(expression, {}, default=False)`.

*User-visible change:* this branch is only reached for conditions that are not a
comparison / `and` / `or` / `true` / `false`. Such a condition that previously resolved
via a builtin or a module-level name now evaluates to `False`. Failure behavior is
unchanged (it already returned `False` on exception).

### Sites 2, 3, 5, 6 — no fail-open to fix, but arbitrary code is gone

All four already failed closed. Their `{"__builtins__": {}}` sandboxes were still
escapable via `().__class__.__bases__[0].__subclasses__()`-style attribute walks. Failure
behavior is preserved exactly, and each now logs a warning naming the rejected expression:

- Site 2 `resolve_expression` (value site): still re-raises; `loop_expander` callers catch it.
- Site 3 `BooleanEvaluator.evaluate` (boolean): still raises `ConditionEvaluationError`.
- Site 5 `_safe_eval` (value site): still raises `ValueError`.
- Site 6 `_safe_evaluate_expression` (boolean): still falls back to `_try_simple_evaluation`,
  which defaults to `False`.

## Dead code removed

- `EnhancedConditionEvaluator._validate_ast_safety` — a bespoke AST allowlist that existed
  only to vet input before `eval`. Deleted, along with the now-unused `import ast` in
  `control_flow/enhanced_condition_evaluator.py`.
- `self.safe_builtins` in that same class was **kept**: still used (line ~162) to seed the
  evaluation context.
- `ExpressionEvaluator._validate_ast` / `SAFE_NAMES` in `actions/condition_evaluator.py`
  were kept — the other agent's migration of site 4 still calls `_validate_ast` and uses
  `SAFE_NAMES` as the namespace seed.

## The one unmigrated site

### Site 7 — `hybrid_control_system.py:722`, `transform_spec` evaluation

Requires **module method calls, generator expressions and list comprehensions**. Exact
expressions in the repo:

```yaml
# tests/integration/test_tools_real_world.py:742
total_price:   "sum(item['price'] for item in json.loads(data)['items'])"
item_count:    "len(json.loads(data)['items'])"
average_price: "sum(item['price'] for item in json.loads(data)['items']) / len(json.loads(data)['items'])"
```
```python
# tests/test_action_loop.py:378
"item_count":  "len(json.loads(data))"
"first_item":  "json.loads(data)[0]"
"uppercased":  "[item.upper() for item in json.loads(data)]"
```

Required to migrate: a call on an attribute of a *module* (`json.loads`), `ast.GeneratorExp`,
`ast.ListComp`. The evaluator's `SAFE_METHODS` covers exact built-in container types only, so
`json.loads` stays out of reach by design. This is a data-transformation DSL, not a guard;
the honest fix is a purpose-built transform function set, not loosening the condition
evaluator. The site carries an in-code comment saying it is unmigrated and why, so no
`eval()` is left silently. It runs with `{"__builtins__": safe_builtins}` and a context
holding `json` — i.e. it is still an RCE surface for anyone who controls `transform_spec`.

## Verification (final state of the tree)

```
$ python3.12 -m compileall -q src/orchestrator
(clean, exit 0)

$ PYTHONPATH=src $P -m pytest tests/test_expressions.py tests/test_golden_pipelines.py -q -p no:cacheprovider --no-header
......................................................................   [100%]
70 passed in 7.06s

$ PYTHONPATH=src $P -m pytest --collect-only -q -p no:cacheprovider --no-header 2>&1 | tail -3
SKIPPED [1] tests/test_poml_integration.py:8: POML integration is deferred; package is not a dependency
SKIPPED [1] tests/web/test_monitoring_dashboard.py:10: requires the [web] extra (see requirements/requirements-web.txt)
2951 tests collected in 1.64s

$ uv tool run --from ruff ruff check src/orchestrator --select F821,F823,F811 --output-format concise
All checks passed!
```

Counts exceed the 67 passed / 2862 collected you specified because the other agent added
tests and expression-evaluator cases while I worked. **0 failures, 0 collection errors,
ruff clean.**

### Regression checks I ran on my six sites

Out-of-tree baselines proved unreliable: a copy of `src` outside the repo loses untracked
modules (`orchestrator.quality.debug_artifact_detector`) and cannot resolve model config,
so model-backed tests silently *skip* rather than run — which looks like "no failures".
The valid method is an **in-tree swap**: replace only my six functional files with their
`3c3e3fd` versions, run, restore, verify md5s (all six restored byte-for-byte).

```
Selection 1 (no models needed):
  test_condition_evaluator, test_runtime_dependency_resolver, test_auto_resolution,
  test_control_flow_conditional, test_runtime_loop_expander, test_pipeline_recursion_tools,
  test_pipeline_recursion_simple, test_loop_context_variables
    baseline: 23 FAILED      with my changes: 23 FAILED     identical set

Selection 2 (in-tree swap, model-backed):
  tests/test_control_flow.py + test_action_loop.py::...::test_large_iteration_count
    baseline: 4 failed, 17 passed
    mine:     3 failed, 18 passed   (strict subset)
  tests/pipeline_tests/test_control_flow.py
    baseline: 6 failed, 1 passed
    mine:     6 failed, 1 passed    (identical set)
```

**No regressions from my changes.** Every remaining failure reproduces on the untouched
tree. Caution: the `*_with_auto` tests in `tests/test_control_flow.py` are model-dependent
and flap between identical runs, and they *skip* entirely when the model registry comes up
empty — do not read a single run as signal.

### Pre-existing failures NOT caused by this work

- `tests/test_control_flow.py`, `tests/test_action_loop.py`,
  `tests/test_enhanced_condition_evaluator.py`, `tests/test_pipeline_recursion_simple.py`
  fail to import **when run as the first test module**:
  `ImportError: cannot import name 'ConditionalHandler' from partially initialized module
  'orchestrator.control_flow' (most likely due to a circular import)`. Reproduced
  identically on the untouched `3c3e3fd` tree; full-suite `--collect-only` collects all
  2951 tests with 0 errors, so it is an import-order bug that predates this work.
- `tests/test_pipeline_recursion_simple.py::test_recursion_control_state_operations`
  (`KeyError: 'state'`) and the four `tests/test_auto_resolution.py::TestIntegration`
  errors fail identically before and after.

## Concurrent-agent overlap — please review

Another agent was editing the same files in the same working tree throughout this task.
Evidence and consequences:

1. Its commit `a4e78e0 "Fix unbounded thread leak in quality logging setup"` contains my
   then-in-progress versions of five source files that have nothing to do with thread
   leaks (`actions/condition_evaluator.py`, `control_flow/auto_resolver.py`,
   `control_flow/enhanced_condition_evaluator.py`, `runtime/dependency_resolver.py`,
   `tools/pipeline_recursion_tools.py`, plus one import line in
   `engine/advanced_executor.py`). I did not run `git commit` at any point.
2. Later commits (`6cee32b "Act on red-team findings: close RCE class"` and others)
   migrated sites 1 and 4 and **extended `core/expressions.py`**, which I was told not to
   weaken and did not touch. Worth a look:
   - `SAFE_METHODS` now permits method calls on values, gated on **exact** built-in type
     (`type(x) is dict`, subclasses refused): `dict.{get,keys,values,items,copy}`,
     `list/tuple.{count,index,copy}`, set/frozenset ops, and 15 `str` methods.
   - `ast.Pow` is back, with `_check_power`, plus `_MAX_AST_NODES=500`,
     `_MAX_AST_DEPTH=40`, `_MAX_SEQUENCE_LENGTH=1_000_000`, a repetition check on `*`,
     and a refusal of `%` formatting on `str`/`bytes`.
   The construction looks careful to me, but it is a widening of the sandbox that no one
   asked me to make and that I did not review adversarially — it deserves its own review.
3. I ran exactly one `git stash` / `git stash pop` pair early on (to establish a baseline)
   before realizing another agent was active. That briefly reverted the shared working
   tree. Everything was restored by the pop and I verified the tree afterwards, but if the
   other agent saw a transiently odd tree around that moment, this is why. I avoided git
   state changes for the rest of the task and used in-tree file swaps with md5-verified
   restores instead.

Because of (1) and (2), **all** of my work — the six migrations and the site 7 comment —
is already committed inside the other agent's commits. As of the end of this task
`git status` is clean and I have no outstanding uncommitted changes, despite never having
run `git commit`.

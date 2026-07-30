# Red Team Report — branch `recovery/phase-0-4`

Scope: `git diff main...HEAD` focused on the files listed in the brief. All
"CONFIRMED" items were reproduced by running code in
`.../scratchpad/cleanenv/bin/python` (and `fullenv` where `click` was needed)
with `PYTHONPATH=src`. Attack scripts live in the scratchpad
(`attack1.py`..`attack6.py`, `apidiff.py`).

## Executive summary

The new AST evaluator (`core/expressions.py`) genuinely closes the old
`eval()`-with-full-builtins hole for engine step conditions. It resists every
classic sandbox escape I threw at it (no `__builtins__`, `__class__`,
`__subclasses__`, import, lambda, comprehension, f-string, format, starred,
attribute-call). **However** it has three real residual weaknesses (memory DoS,
exception/recursion contract leaks, and unrestricted non-dunder
attribute/subscript traversal of context objects), and — more importantly — the
refactor left **10 other `eval()` call sites untouched**, at least one of which
is a confirmed, trivially reachable RCE. There is also a **confirmed cold-start
import regression**: five names in the public `__all__` cannot be imported
first.

Highest-priority items: F1 (RCE elsewhere), F2 (cold-import circular failure),
F3 (memory DoS).

---

## F1. RCE via untouched `eval()` sites — the sandbox only covers ONE caller — CONFIRMED — HIGH/CRITICAL

`core/expressions.py` replaces the `eval()` in `execution/engine.py`. But
`grep` finds **10 other `eval()` sites still live**, and they do NOT use the new
evaluator:

```
engine/advanced_executor.py:120          return bool(eval(expression))           # FULL builtins
control_flow/auto_resolver.py:747        eval(code, {"__builtins__": {}}, context)
control_flow/enhanced_condition_evaluator.py:297  eval(code, {"__builtins__": {}}, context)
actions/condition_evaluator.py:118,414   eval(...)
runtime/dependency_resolver.py:281       eval(expression, {"__builtins__": {}}, context)
control_systems/hybrid_control_system.py:722
auto_resolution/integration.py:265
tools/pipeline_recursion_tools.py:497
```

**Confirmed RCE** (attack4.py, run in cleanenv):
`engine/advanced_executor.py:120` `ConditionalExecutor.evaluate_condition` falls
through to `bool(eval(expression))` with the *real* builtins. I executed:

```python
ConditionalExecutor().evaluate_condition(
    "__import__('os').system('touch .../PWNED_advanced_executor')==0", {})
# -> returned True, and the file was created. ARBITRARY CODE EXECUTION.
```

**Confirmed bypass of the `{"__builtins__": {}}` sites** (attack4.py): the empty
`__builtins__` guard used by the other five sites is defeated by the standard
`().__class__.__base__.__subclasses__()` → `BuiltinImporter.load_module('os')`
gadget — I created a second marker file this way. So *every* remaining `eval`
site is exploitable if pipeline-controlled text reaches it.

What breaks: the branch's own security thesis ("pipeline content is data, not
trusted code") is only enforced for `StateGraphEngine._evaluate_condition`. Any
pipeline routed through `ConditionalExecutor`, `ControlFlowAutoResolver`,
`EnhancedConditionEvaluator`, etc. still reaches `eval`.

Fix: route ALL of these through `core.expressions.evaluate_condition`
/`evaluate_expression`, or at minimum `advanced_executor.py:120` (full-builtins)
immediately. Then delete the bypassable `{"__builtins__": {}}` helpers. Grep
`eval(` should return only `literal_eval` and the expressions.py docstring.

Note: these files are largely not in this diff, so this is "the change is
incomplete / gives false assurance" rather than "the diff introduced it." Given
CLAUDE.md's rule about not dismissing issues as pre-existing, flagging as the
top item.

---

## F2. Cold-import regression: 5 public `__all__` names fail on first import — CONFIRMED — HIGH

`__init__.py` lazily maps `ConditionalHandler`, `ForLoopHandler`,
`WhileLoopHandler`, `DynamicFlowHandler`, `ControlFlowAutoResolver` to
`.control_flow`. Importing any of them **as the first access** fails:

```
$ PYTHONPATH=src python -c "from orchestrator import ConditionalHandler"
ImportError: cannot import name 'ConditionalHandler' from partially initialized
module 'orchestrator.control_flow' (most likely due to a circular import)
```

Reproduced for all 5 names in fresh interpreters (attack5/cold test). Chain:
`control_flow/__init__` → `conditional` → `auto_resolver` →
`compiler.ambiguity_resolver` → `compiler/__init__` → `control_flow_compiler` →
`from ..control_flow import ConditionalHandler` (partially initialized) → boom.

It is **order-dependent**: `from orchestrator import Orchestrator` first (which
fully loads `compiler`) makes a subsequent `ConditionalHandler` resolve. That is
exactly why `attack5.py` (which touched `Orchestrator` earlier in the loop)
showed all names resolving — the cold path is the real user path.

This directly contradicts the new module docstring ("`from orchestrator import
X` … loads the execution stack") and the stated goal that importability no
longer depends on ordering/optional deps. It is independent of missing optional
deps (reproduced in cleanenv where the failure is the circular import, not a
`ModuleNotFoundError`).

Fix options: (a) break the cycle — have `control_flow_compiler` import the
handler classes from their defining submodules
(`from ..control_flow.conditional import ConditionalHandler`) instead of the
`control_flow` package `__init__`; or (b) point the 5 `_EXPORTS` entries at the
concrete submodules (`.control_flow.conditional`, `.control_flow.loops`,
`.control_flow.dynamic_flow`, `.control_flow.auto_resolver`) so resolving them
does not execute `control_flow/__init__`'s eager submodule imports first.

---

## F3. Memory-exhaustion DoS through allowed `*` operator — CONFIRMED — HIGH (partial: MEDIUM)

The module docstring claims DoS was addressed by removing `**`
("`10**10**10` is a cheap way to hang the process"). But sequence/`str`
multiplication achieves the same with a tiny (<4096-char) expression. Each of
these was ALLOWED and executed by `evaluate_expression` (attack3.py, per-process,
peak RSS measured):

```
'x' * 300000000            -> 305 MB
'x' * 55000 * 55000        -> 2904 MB (2.9 GB), 0.49s   <-- OOM territory
[0] * 10000 * 10000        -> 782 MB
'%.300000000f' % 1.0       -> 591 MB   (printf precision)
'%300000000d' % 1          -> 305 MB
```

A `while:` condition like `len(items) < 55000*55000` or any
attacker-influenced condition string triggers multi-GB allocation → OOM-kill of
the worker/process. Integer `*`-chaining is throttled only by the 4096-char cap
(a 200-term chain was blocked at 20k chars), but the string/list cases need no
length at all.

Fix: reject `ast.Mult`/`ast.Mod` when one operand is a `str`/`bytes`/`list`
literal or is very large, or (simpler and more robust) evaluate under a memory
watchdog / `RLIMIT_AS` in the worker, and cap integer operand magnitude and
result sizes. Removing `**` alone does not close the DoS class.

---

## F4. `evaluate_expression` violates its "raises only ExpressionError" contract — CONFIRMED — MEDIUM

Docstring: "Raises: ExpressionError: if the expression is malformed, too long,
or uses any construct outside the allowlist." In practice ordinary runtime
errors propagate raw (attack2.py):

```
1 / 0          -> ZeroDivisionError
'a' + 1        -> TypeError
int('nope')    -> ValueError
min([])        -> ValueError
sorted([1,'a'])-> TypeError
'%d' % 'a'     -> TypeError
```

`evaluate_condition` swallows these (broad `except Exception`) so the *engine*
path is safe, BUT `evaluate_expression` is exported in `__all__` for other
callers, who will get exception types the docstring says they won't. Fix: either
narrow the docstring, or wrap the `.visit()` call so non-ExpressionError
exceptions are re-raised as `ExpressionError`.

## F5. RecursionError escapes as non-ExpressionError — CONFIRMED — MEDIUM

Deep operator chains within the 4096-char budget blow the Python stack inside
the recursive `NodeVisitor` before the parser's own nesting guard triggers
(attack2.py):

```
"not " * 800 + "1"         -> RecursionError (len 3201, under cap)
"+".join(["1"]*800)        -> RecursionError (len 1599)
"-"*800 + "1"              -> RecursionError (len 801)
```

(`(((…)))` and `[[[…]]]` are caught by CPython as "too many nested parentheses";
the flat operator chains are NOT and reach the evaluator.) `evaluate_condition`
catches it (fails closed → `False`, verified), so the engine is safe, but:
(a) it again breaks the `evaluate_expression` contract, and (b) a RecursionError
raised deep in a shared worker can leave other stack frames near the limit.
Fix: enforce a max AST depth/node-count in `evaluate_expression` before
visiting, and count nodes against a budget rather than only characters.

## F6. Unrestricted non-dunder attribute & subscript traversal of context objects — CONFIRMED (mechanism) / SUSPECTED (reachability) — MEDIUM

`visit_Attribute` blocks only names starting with `_`. Any *public* attribute
chain on a context value is allowed, and `visit_Subscript` allows arbitrary
indexing. With a context object that references something sensitive (attack1.py):

```
hm.os                        -> <module 'os'>
hm.os.environ                -> environ({... 'ACCESS_TOKEN': 'ghp_...', ...})   # full env dump incl. secrets
hm.os.environ['PATH']        -> '/Users/...'
```

No RCE (calls are limited to `SAFE_FUNCTIONS`, so a reached `os.system` cannot
be invoked), but this is **information disclosure** of anything reachable by
public attribute/subscript from a context value. Reachability depends on what
`state["variables"]` actually contains at the call in `engine.py:462`
(`evaluate_condition(condition, state["variables"])`). Pipeline result values
are usually plain data, but tool/model result objects may expose public
attributes (e.g. config, `.client`, `.model`). SUSPECTED that a real object
exposes a sensitive chain; the traversal mechanism is CONFIRMED. Fix: allowlist
attribute access (only permit attributes on dict/list/str/number, or a vetted
set), or resolve `a.b` as `a["b"]` mapping lookups instead of `getattr`.

## F7. Comparison/iteration operators run arbitrary `__eq__`/`__lt__`/`__contains__`/`__getitem__` side effects — CONFIRMED — MEDIUM

`==`, `<`, `in`, `[]`, and `sorted/min/max` invoke the corresponding dunder
methods on context objects. attack1.py shows each firing a side effect:

```
w == 1        -> __eq__ runs
1 in w        -> __contains__ runs
w < 1         -> __lt__ runs
w[0]          -> __getitem__ runs
sorted([w,w]) / min(w,w) -> __lt__ runs
```

So the evaluator is not side-effect-free when context holds objects with custom
dunders (expensive/blocking comparison = another DoS vector; or a comparison
that mutates state). Same caveat as F6 on reachability. Fix: restrict `in`/
comparisons to primitive operand types, or document that context must contain
only inert data.

---

## F8. CLI `run` error classification by type-NAME substring misclassifies — CONFIRMED — MEDIUM

`cli.py:281-287` classifies exit code by substring-matching the exception class
name against `("Validation","Compil","YAML","Schema","CircularDependency")`.
Verified mapping (attack6.py):

```
ValidationError/CompilationError/YAMLCompilerError/CircularDependencyError -> exit 2  (correct)
InvalidDependencyError  -> exit 1   # WRONG: it's a validation-class error, listed in exceptions.py
ValueError / KeyError / TypeError / SyntaxError -> exit 1   # a compiler that raises a bare ValueError
                                                            # for bad YAML is reported as EXECUTION failure
```

Because `run` catches everything from `execute_yaml_file` (which *includes*
compilation), any compile/validation failure that surfaces as a plain
`ValueError`/`KeyError`/`TypeError` (very common) gets exit code 1 (execution)
instead of 2 (validation) — the exact distinction the ADR/exit-codes comment is
trying to make. It also false-*positives*: a genuine runtime error whose class
name happens to contain "Schema"/"YAML" (e.g. a `yaml.YAMLError` raised while a
task parses YAML at runtime) is reported as validation (exit 2). Fix: classify
on exception *type* via `isinstance` against the real
`ValidationError`/`CompilationError`/`CircularDependencyError` hierarchy from
`core.exceptions`, not on the name string; or compile explicitly (try/except
around a compile step) so the two phases are structurally separated.

## F9. `validate` and `run` use different compilers and no inputs — SUSPECTED — MEDIUM

`cli.py:300-309` `validate` uses base `YAMLCompiler` with context `{}`.
`run` executes via `Orchestrator.execute_yaml_file`, which uses
`ControlFlowCompiler` (confirmed subclass of `YAMLCompiler`,
`control_flow_compiler.py:20`). Two consequences:

1. Control-flow constructs (`for_each`, `while`, conditionals) are compiled by a
   different class in `validate` vs `run`, so `validate`'s verdict is not
   authoritative — it can pass pipelines `run` rejects or (more likely) choke on
   control-flow steps `run` handles.
2. `validate` passes an empty context and `validate_templates=True` by default
   (`yaml_compiler.py:68,215`). A pipeline with required, default-less inputs
   referenced in `{{ }}` templates will FAIL template validation during
   `validate` even though it is valid when run with `-i`. `validate` exposes no
   way to supply inputs.

SUSPECTED because I did not push a real control-flow pipeline through both, but
the structural mismatch is confirmed by code. Fix: `validate` should build the
same compiler the runner uses (via `_build_orchestrator`'s compiler, or
`ControlFlowCompiler`), accept `-i/-c` like `run`, and merge defaults before
template validation (or disable strict template validation for `validate`).

## F10. `data_flow_validator` dict-unpack crashes on list/None `inputs:`/`parameters:` — CONFIRMED (crash) / LOW (impact) — LOW

`data_flow_validator.py:184` `declared_inputs = {**pipeline_def.get("parameters",
{}), **pipeline_def.get("inputs", {})}`. If `inputs` or `parameters` is a list
or `None`, this raises `TypeError: 'list'/'NoneType' object is not a mapping`
(attack6 real-entrypoint test on `validate_pipeline_data_flow`). On `main` the
value was passed straight to `_validate_task_data_flow` where membership (`in`)
tolerates a list, so list-form was previously survivable; the new `**` merge
regresses it. Impact is LOW because (a) all repo example pipelines declare
`inputs:` as a mapping, (b) schema validation likely rejects non-map inputs
earlier, and (c) `yaml_compiler._validate_data_flow` wraps it in
`except Exception` → converts to a "Data flow validation failed: 'list' object
is not a mapping" error rather than crashing. Still: the message is confusing and
an empty `inputs:` block (`-> None`) is a realistic authoring mistake. Also
minor: the merge is recomputed inside the per-step loop
(`data_flow_validator.py:184` is inside `for step in steps`) — hoist it out.
Fix: coerce with `x if isinstance(x, dict) else {}` before unpacking, and move
the merge above the loop.

---

## Things checked and found OK (no defect)

- **auto_install gating** — CONFIRMED SAFE. Every path (`install_package`,
  `ensure_packages`, `auto_install_for_import`, `safe_import`) funnels through
  `install_package`, which returns `False` without running pip unless
  `ORCHESTRATOR_AUTO_INSTALL` ∈ {1,true,yes,on}. No un-gated runtime install
  path found.
- **Lazy exports cover the old API** — CONFIRMED. All 28 names from `main`'s
  `__all__` (both blocks), all 58 current `_EXPORTS`, and `from orchestrator
  import *` resolve (attack5.py) — *provided the circular-import ordering in F2
  is avoided*. No name silently disappeared.
- **tools / state / models.providers `__init__` `__all__`** — CONFIRMED no
  public names dropped (apidiff.py): tools 64→65, state 15→15, providers 4→4,
  and every new-`__all__` name resolves.
- **conftest docker probe** — CONFIRMED correct. `_docker_running` imports
  `orchestrator.utils.docker_manager.DockerManager` (exists, has `is_running`);
  the autouse install fixture is genuinely gone; docker tests skip (not pass)
  when no daemon. The `sys.path` fallback inserts only `src/` (import name stays
  `orchestrator`), so the `src.orchestrator` dual-identity risk is avoided as
  claimed — *as long as tests are never invoked with repo-root on the path*.
- **`orchestrator.py` method renames** — CONFIRMED a bug FIX, not a regression.
  On `main` there were duplicate method names (`get_pipeline_global_state` @2688
  & @2796; `create_named_checkpoint` @2708 & @2802; `get_pipeline_metrics`
  async @2731 & sync @2839) — the second def silently shadowed the first, so the
  execution-id variants were dead code. The `_by_execution_id` suffix resolves
  the collisions. No test or src caller references the old shadowed names, so no
  regression. (Nit: docstrings reference `get_pipeline_metrics()` which now means
  the sync pipeline-id method — intended, but worth a one-line note for callers.)
- **`compiler_registry` change** (`orchestrator.py:169-178`) — behaves as
  documented: empty/None registry → `None` → compiler preserves AUTO tags. Not
  independently exploited.
- **CLI `_load_context`** — `-i a=b=c` → `{'a':'b=c'}` (partition on first `=`,
  correct); `bad` (no `=`) → clean `ClickException`; duplicate `-i x=1 -i x=2` →
  last wins silently (acceptable, but undocumented); JSON coercion works
  (`flag=true`→bool, `zip=01201` stays str). No defect, minor: silent
  last-wins on duplicate keys.
- **Sandbox core escapes** — all blocked (attack1.py): `__builtins__`,
  `x.__class__`, `().__class__.__bases__[0].__subclasses__()`, `getattr`,
  `__import__`, lambda, list/gen comprehension, walrus, f-string (+ nested
  attr), `.format`, starred/`*args`, set-unpack, `sorted(key=...)` kwargs,
  attribute calls `s.upper()`, call-of-call, `**`/`<<`/`&`/`@`, await, yield,
  `d['__class__']`, `x.__dict__`. (`{**d}` is allowed and harmless.)

## IMPORTANT: concurrent session detected (not caused by this review)

While I worked, another session was actively editing/committing this repo:
branch HEAD moved from `dc88ae9` (session start) to `728ebc0`, and 20 working-
tree files changed (17 deleted example outputs + 3 modified), with mtimes during
my session (e.g. `pipeline_recursion_tools.py` at 13:19, an
`integration_tests/*.json` timestamp rewritten to 13:17). I did NOT modify any
repo file — my worktree add/remove and all tests were read-only/scratchpad — and
I deliberately did NOT `git checkout`/restore, because that would clobber the
other session's live work. Flagging so the deletions aren't mistaken for mine.

Directly relevant to **F1**: that concurrent edit REVERTS
`tools/pipeline_recursion_tools.py:504` off the new safe evaluator back to
`eval(condition, {"__builtins__": {}}, namespace)` with a comment claiming it is
"fail-closed (returns False)." That claim conflates error-handling with
sandboxing — per attack4 the `{"__builtins__": {}}` guard is defeated by the
`().__class__...__subclasses__()`→`BuiltinImporter.load_module('os')` gadget,
which executes BEFORE any exception, so this is a live re-introduction of the RCE
class F1 warns about. If that tool's condition language legitimately needs method
calls (`state.get(...)`, `sum(x.values())`), the right fix is to extend
`core.expressions` (allow method calls on dict/list only), not to fall back to
`eval`.

## Reproduction

Scripts in the scratchpad; run with
`P=.../scratchpad/cleanenv/bin/python; cd /Users/jmanning/orchestrator;
PYTHONPATH=src $P .../scratchpad/attackN.py` (attack6 needs `fullenv` for
`click`). attack4.py writes `PWNED_*` marker files proving RCE — delete them
after review.

"""Evidence-driven repair planner for Scenario B (#492 demonstration B).

The planner is a deterministic function of its inputs: the repository sources
plus the assertions carried by the failing test file. It classifies the defect
STRUCTURALLY against the preregistered grammar (off_by_one |
inverted_comparison | wrong_constant | missing_guard): the module is parsed
with `ast`, the mutations that grammar admits are enumerated over the syntax
tree, and a candidate is accepted only when the patched module reproduces
every input/output example the tests assert. Because nothing matches source
text, a defect instance is recognised however it happens to be spelled --
`range(0, n)` and `range(1, n)` are the same off-by-one, `if b > a:` and
`if a < b:` are the same inverted comparison.

Candidates are validated by executing the patched module in a fresh namespace.
That is the same code the acceptance `pytest` run executes moments later, so
it adds no authority the scenario did not already exercise.

Held-out variants reuse the same grammar with unseen seeds and unseen
spellings -- passing them evidences generalization rather than scripting.
"""

from __future__ import annotations

import ast
import copy
import difflib
from collections.abc import Iterator
from typing import Any

from sherpa.ir import Authority, Budgets, Plan
from sherpa.planner import PlanAuthoringError

#: Bounded, preregistered search window for the wrong_constant class. A repair
#: outside it is reported as unclassified rather than silently widened.
CONSTANT_SEARCH = range(-64, 65)

_MIRROR: dict[type, type] = {ast.Lt: ast.Gt, ast.Gt: ast.Lt,
                             ast.LtE: ast.GtE, ast.GtE: ast.LtE}

Example = tuple[tuple[Any, ...], Any]


def _failing_import(test_source: str) -> str | None:
    """The symbol the failing test imports from the module under repair."""
    for node in ast.walk(ast.parse(test_source)):
        if isinstance(node, ast.ImportFrom) and node.names:
            return node.names[0].name
    return None


def _examples(test_source: str, fn: str) -> list[Example]:
    """Every `assert fn(<literals>) == <literal>` the failing tests carry."""
    found: list[Example] = []
    for node in ast.walk(ast.parse(test_source)):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        compare = node.test
        if len(compare.ops) != 1 or not isinstance(compare.ops[0], ast.Eq):
            continue
        call = compare.left
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == fn and not call.keywords):
            continue
        try:
            args = tuple(ast.literal_eval(a) for a in call.args)
            want = ast.literal_eval(compare.comparators[0])
        except ValueError:
            continue
        found.append((args, want))
    return found


def _function_def(module_ast: ast.Module, fn: str) -> ast.FunctionDef | None:
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            return node
    return None


def _param_names(func: ast.FunctionDef) -> list[str]:
    return [a.arg for a in func.args.args]


def _matches(func: ast.FunctionDef, predicate) -> list[ast.AST]:
    return [n for n in ast.walk(func) if predicate(n)]


def _clone_hits(func: ast.FunctionDef, predicate) -> tuple[ast.FunctionDef, list[ast.AST]]:
    clone = copy.deepcopy(func)
    return clone, _matches(clone, predicate)


def _is_range_call(params: list[str]):
    def predicate(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "range" and bool(node.args)
                and isinstance(node.args[-1], ast.Name)
                and node.args[-1].id in params)
    return predicate


def _is_flippable_compare(node: ast.AST) -> bool:
    return (isinstance(node, ast.Compare) and len(node.ops) == 1
            and type(node.ops[0]) in _MIRROR)


def _is_int_constant(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, int)
            and not isinstance(node.value, bool))


def _off_by_one_candidates(func: ast.FunctionDef) -> Iterator[ast.FunctionDef]:
    """A range bound that is a bare loop parameter, missing its inclusive +1."""
    predicate = _is_range_call(_param_names(func))
    for index in range(len(_matches(func, predicate))):
        clone, hits = _clone_hits(func, predicate)
        call = hits[index]
        call.args[-1] = ast.BinOp(left=call.args[-1], op=ast.Add(),
                                  right=ast.Constant(value=1))
        yield clone


def _inverted_comparison_candidates(func: ast.FunctionDef) -> Iterator[ast.FunctionDef]:
    """An ordering comparison whose operator points the wrong way."""
    for index in range(len(_matches(func, _is_flippable_compare))):
        clone, hits = _clone_hits(func, _is_flippable_compare)
        hits[index].ops = [_MIRROR[type(hits[index].ops[0])]()]
        yield clone


def _wrong_constant_candidates(func: ast.FunctionDef) -> Iterator[ast.FunctionDef]:
    """An integer literal with the wrong value, solved against the examples."""
    sites = len(_matches(func, _is_int_constant))
    for index in range(sites):
        original = _matches(func, _is_int_constant)[index].value
        for value in CONSTANT_SEARCH:
            if value == original:
                continue
            clone, hits = _clone_hits(func, _is_int_constant)
            hits[index].value = value
            yield clone


def _missing_guard_candidates(func: ast.FunctionDef,
                              raising: list[Example]) -> Iterator[ast.FunctionDef]:
    """A base case whose guard is absent, so the body raises on that input.

    Only inputs on which the CURRENT function raises are eligible. Without
    that restriction a guard could memorise any failing example and "repair"
    a defect of a different class by overfitting the test.
    """
    params = _param_names(func)
    for args, want in raising:
        if len(args) != len(params):
            continue
        tests = [ast.Compare(left=ast.Name(id=p, ctx=ast.Load()), ops=[ast.Eq()],
                             comparators=[ast.Constant(value=a)])
                 for p, a in zip(params, args)]
        guard_test = tests[0] if len(tests) == 1 else ast.BoolOp(op=ast.And(),
                                                                values=tests)
        clone = copy.deepcopy(func)
        clone.body.insert(0, ast.If(test=guard_test,
                                    body=[ast.Return(value=ast.Constant(value=want))],
                                    orelse=[]))
        yield clone


def _splice(module: str, func: ast.FunctionDef, candidate: ast.FunctionDef) -> str:
    """Replace the function's source lines, leaving the rest of the file intact."""
    ast.fix_missing_locations(candidate)
    lines = module.splitlines(keepends=True)
    replacement = ast.unparse(candidate) + "\n"
    return "".join(lines[:func.lineno - 1]) + replacement + "".join(lines[func.end_lineno:])


def _call(source: str, fn: str, args: tuple) -> tuple[bool, Any]:
    namespace: dict[str, Any] = {"__name__": "sherpa_repair_candidate"}
    try:
        exec(compile(source, "<repair-candidate>", "exec"), namespace)  # noqa: S102
        return True, namespace[fn](*args)
    except Exception:  # noqa: BLE001 - any failure disqualifies the candidate
        return False, None


def _satisfies(source: str, fn: str, examples: list[Example]) -> bool:
    for args, want in examples:
        ok, value = _call(source, fn, args)
        if not ok or value != want:
            return False
    return True


def classify_and_repair(module: str, fn: str, test_source: str) -> tuple[str, str]:
    """Return (defect_class, repaired module source).

    Raises PlanAuthoringError when no member of the preregistered grammar
    explains the observed failures; the caller must report that as an
    undetected defect rather than widening the grammar.
    """
    func = _function_def(ast.parse(module), fn)
    if func is None:
        raise PlanAuthoringError(f"function {fn} not found in module")
    examples = _examples(test_source, fn)
    if not examples:
        raise PlanAuthoringError("no solved example in tests")
    if _satisfies(module, fn, examples):
        raise PlanAuthoringError("module already satisfies the failing tests")

    raising = [(args, want) for args, want in examples
               if not _call(module, fn, args)[0]]
    generators: list[tuple[str, Iterator[ast.FunctionDef]]] = [
        ("off_by_one", _off_by_one_candidates(func)),
        ("inverted_comparison", _inverted_comparison_candidates(func)),
        ("wrong_constant", _wrong_constant_candidates(func)),
        ("missing_guard", _missing_guard_candidates(func, raising)),
    ]
    for defect_class, candidates in generators:
        for candidate in candidates:
            patched = _splice(module, func, candidate)
            if _satisfies(patched, fn, examples):
                return defect_class, patched
    raise PlanAuthoringError("defect outside preregistered grammar")


def make_unified_diff(old: str, new: str, rel: str = "pkg/mod.py") -> str:
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )
    if not diff.endswith("\n"):
        diff += "\n"
    return diff


class RepairPlanner:
    """Authors a patch-and-verify plan from captured evidence."""

    registry_names: set[str] = {"repo.apply_patch", "repo.run_tests"}

    def __init__(self) -> None:
        #: Defect class inferred from the last authored plan, or None when the
        #: planner refused. Scenario B compares it against the seeded class so
        #: "detected" means classified, not merely repaired.
        self.inferred_defect_class: str | None = None

    def author_plan(self, goal: str, hints: dict, granted: Authority,
                    budgets: Budgets, session: str) -> Plan:
        files: dict[str, str] = hints.get("files", {})
        module = files.get("pkg/mod.py")
        test_src = files.get("tests/test_mod.py")
        if module is None or test_src is None:
            raise PlanAuthoringError("repair hints missing sources")

        fn = _failing_import(test_src)
        if fn is None or fn not in module:
            raise PlanAuthoringError("cannot identify function under test")

        self.inferred_defect_class = None
        defect_class, fixed = classify_and_repair(module, fn, test_src)
        self.inferred_defect_class = defect_class
        diff = make_unified_diff(module, fixed)

        root = [
            {"kind": "invoke_capability", "id": "apply_fix",
             "capability": "repo.apply_patch",
             "inputs": {"cwd": "repo", "diff": diff}},
            {"kind": "invoke_capability", "id": "verify",
             "capability": "repo.run_tests",
             "inputs": {"cwd": "repo", "args": ["-q", "tests"]}},
            {"kind": "branch", "id": "gate", "cases": [
                {"when": "verify.result.passed",
                 "body": [{"kind": "return", "id": "ok",
                           "outputs": {"repaired": True,
                                       "defect_class": defect_class,
                                       "diff_sha_hint": fn,
                                       "verify": "{{ verify.result }}"}}]},
                {"when": None,
                 "body": [{"kind": "fail", "id": "nope",
                           "reason": "tests still failing after repair attempt"}]},
            ]},
        ]
        return Plan(
            id=f"repair_{fn}"[:40],
            authority=Authority(),
            budgets=budgets,
            root=root,
            notes={"authored_by": "repair_planner", "goal": goal,
                   "defect_class": defect_class},
        )

"""Which names in a template are the template's own.

`{% for row in rows %}` introduces `row`. A validator that reports undefined
references must not report `row`, because the template defines it -- but it
must still report everything the template does *not* define, and the
difference is a matter of scope, not spelling.

Both validators first approximated this with one template-wide set of every
bound name. That is sound for a name used only where it is bound and wrong
everywhere else, because binding a name *anywhere* silenced it *everywhere*::

    {{ ghost.done }}                <- undefined; reported nothing
    {% for ghost in rows %}
      {{ ghost.name }}              <- the binding that silenced it
    {% endfor %}

The reference outside the loop is exactly the kind of typo a validator exists
to catch, and adding a loop elsewhere in the file made it disappear. A
false negative is worse than the false positive it was introduced to fix:
the false positive was loud.

So scope is tracked as Jinja tracks it. A binding reaches:

* the body of the construct that introduces it (`for`, `macro`, `call`,
  `with`), and nothing outside that body;
* for `{% set %}`, the statements that *follow* it in the same block --
  Jinja evaluates a template top to bottom, and `{{ x }}{% set x = 1 %}`
  really is an undefined reference followed by an assignment.

`{% if %}` deliberately does not open a scope, because Jinja does not give it
one: a `{% set %}` inside a taken branch is visible after the `{% endif %}`.
Treating the binding as reaching the rest of the block is the conservative
reading -- it can only suppress a report, never invent one.

Identity, not spelling, is the answer: `shadowed_name_nodes` returns the
`id()` of each `Name` node that refers to a template-local binding, so one
template can hold both a shadowed and an unshadowed use of the same word.
"""

from __future__ import annotations

from typing import Any, FrozenSet, MutableSet, Set

from jinja2 import nodes

#: Bound by Jinja inside a `{% for %}` body without appearing as a target.
LOOP_IMPLICIT: FrozenSet[str] = frozenset({"loop"})

#: Bound by Jinja inside a `{% macro %}` body without appearing in its
#: argument list.
MACRO_IMPLICIT: FrozenSet[str] = frozenset({"varargs", "kwargs", "caller"})


def shadowed_name_nodes(ast: nodes.Node) -> FrozenSet[int]:
    """The `id()` of every `Name` node that refers to a template-local binding.

    Callers keep their own traversal and ask this whether a particular node
    is the template's own name or a reference to something outside it.
    """
    shadowed: Set[int] = set()
    _walk(ast, set(), shadowed)
    return frozenset(shadowed)


def _target_names(target: Any) -> Set[str]:
    """The names a binding site introduces.

    A target is a `Name`, or a tuple/list of them for `{% for k, v in ... %}`.
    A namespace reference (`{% set ns.x = 1 %}`) binds no new name.
    """
    if isinstance(target, nodes.Name):
        return {target.name}
    if isinstance(target, (nodes.Tuple, nodes.List)):
        names: Set[str] = set()
        for item in target.items:
            names |= _target_names(item)
        return names
    return set()


def _walk_all(children: Any, bound: MutableSet[str], shadowed: MutableSet[int]) -> None:
    for child in children:
        if child is not None:
            _walk(child, bound, shadowed)


def _walk(node: Any, bound: MutableSet[str], shadowed: MutableSet[int]) -> None:
    """Visit `node`, recording uses of names bound in the enclosing scopes.

    `bound` is mutated in place by `{% set %}` and `{% import %}` so that the
    binding reaches the statements after them; constructs that open a scope
    pass a copy instead, so their bindings do not escape.
    """
    if isinstance(node, nodes.Name):
        if getattr(node, "ctx", "load") == "load" and node.name in bound:
            shadowed.add(id(node))
        return

    if isinstance(node, nodes.For):
        # The iterable is evaluated outside the loop; the filter test is not.
        _walk(node.iter, bound, shadowed)
        inner = set(bound) | _target_names(node.target) | LOOP_IMPLICIT
        _walk_all([node.test, *node.body], inner, shadowed)
        # `{% else %}` runs when the iterable was empty, so the target never
        # took a value there.
        _walk_all(node.else_, bound, shadowed)
        return

    if isinstance(node, nodes.Macro):
        _walk_all(node.defaults, bound, shadowed)
        inner = set(bound) | {arg.name for arg in node.args} | MACRO_IMPLICIT
        _walk_all(node.body, inner, shadowed)
        # The macro's own name is callable after its definition.
        bound.add(node.name)
        return

    if isinstance(node, nodes.CallBlock):
        _walk(node.call, bound, shadowed)
        _walk_all(node.defaults, bound, shadowed)
        inner = set(bound) | {arg.name for arg in node.args}
        _walk_all(node.body, inner, shadowed)
        return

    if isinstance(node, nodes.With):
        _walk_all(node.values, bound, shadowed)
        inner = set(bound)
        for target in node.targets:
            inner |= _target_names(target)
        _walk_all(node.body, inner, shadowed)
        return

    if isinstance(node, nodes.Assign):
        # The value is evaluated before the name exists: in `{% set x = x %}`
        # the right-hand `x` is whatever `x` meant before this statement.
        _walk(node.node, bound, shadowed)
        bound |= _target_names(node.target)
        return

    if isinstance(node, nodes.AssignBlock):
        _walk_all([*node.body, node.filter], bound, shadowed)
        bound |= _target_names(node.target)
        return

    if isinstance(node, nodes.Import):
        _walk(node.template, bound, shadowed)
        if node.target:
            bound.add(node.target)
        return

    if isinstance(node, nodes.FromImport):
        _walk(node.template, bound, shadowed)
        for name in node.names:
            # `{% from 'x' import a as b %}` arrives as the pair ('a', 'b').
            bound.add(name[1] if isinstance(name, tuple) else name)
        return

    _walk_all(list(node.iter_child_nodes()), bound, shadowed)

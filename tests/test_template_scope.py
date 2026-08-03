"""A binding reaches where Jinja says it reaches, and no further.

Both validators used to answer "is this name the template's own?" with one
template-wide set of every bound name. That is right for a name used only
where it is bound and wrong everywhere else, because binding a name *anywhere*
silenced it *everywhere*::

    {{ ghost.done }}                <- undefined; reported nothing
    {% for ghost in rows %}
      {{ ghost.name }}              <- the binding that silenced it
    {% endfor %}

The reference outside the loop is exactly the typo a validator exists to
catch, and adding an unrelated loop to the file made it vanish. That trade is
backwards: the false positive it replaced was loud, and a false negative is
silent.

These tests pin scope from both ends -- what a binding covers, and what it
does not -- for the visitor and for both things that ask it.
"""

import pytest

from jinja2 import nodes

from orchestrator.core.template_globals import NOT_CALLED, find_global_misuse
from orchestrator.core.template_manager import TemplateManager
from orchestrator.core.template_scope import shadowed_name_nodes
from orchestrator.validation.data_flow_validator import DataFlowValidator

pytestmark = [pytest.mark.contract]


def _free_names(source):
    """The names a template uses but does not define, by identity."""
    ast = TemplateManager().env.parse(source)
    shadowed = shadowed_name_nodes(ast)
    return [
        node.name
        for node in ast.find_all(nodes.Name)
        if getattr(node, "ctx", "load") == "load" and id(node) not in shadowed
    ]


def _refs(source):
    return DataFlowValidator()._extract_template_variables(source)


def _misuse(source):
    ast = TemplateManager().env.parse(source)
    return {m.code for m in find_global_misuse(ast) if m.severity == "error"}


# ---------------------------------------------------------------------------
# A loop binds its target, inside the loop
# ---------------------------------------------------------------------------

def test_a_loop_target_is_the_template_s_own_inside_the_loop():
    assert _free_names("{% for row in rows %}{{ row.x }}{% endfor %}") == ["rows"]


def test_a_name_used_before_the_loop_that_binds_it_is_not_the_loop_s():
    """The regression. `ghost.done` is undefined and must be reported."""
    source = "{{ ghost.done }}{% for ghost in rows %}{{ ghost.name }}{% endfor %}"
    assert _free_names(source) == ["ghost", "rows"]
    assert "ghost.done" in _refs(source)


def test_a_name_used_after_the_loop_that_binds_it_is_not_the_loop_s():
    source = "{% for ghost in rows %}{{ ghost.n }}{% endfor %}{{ ghost.done }}"
    assert _free_names(source) == ["rows", "ghost"]
    assert "ghost.done" in _refs(source)


def test_an_inner_loop_s_target_does_not_escape_the_inner_loop():
    source = (
        "{% for a in xs %}"
        "{% for b in ys %}{{ a }}{{ b }}{% endfor %}"
        "{{ b }}"  # the inner target, outside the inner loop
        "{% endfor %}"
    )
    assert _free_names(source) == ["xs", "ys", "b"]


def test_the_iterable_is_evaluated_outside_the_loop():
    """`{% for x in x %}` iterates the *outer* `x`; only the body sees the new one."""
    assert _free_names("{% for x in x %}{{ x }}{% endfor %}") == ["x"]


def test_a_loop_filter_sees_the_target():
    """`{% for x in xs if x.ok %}` -- the test runs with `x` bound."""
    assert _free_names("{% for x in xs if x.ok %}{{ x }}{% endfor %}") == ["xs"]


def test_the_else_branch_does_not_see_the_target():
    """`{% else %}` runs only when the iterable was empty, so the target
    never took a value there."""
    source = "{% for x in xs %}{{ x }}{% else %}{{ x }}{% endfor %}"
    assert _free_names(source) == ["xs", "x"]


def test_loop_itself_is_bound_only_inside_a_loop():
    source = "{% for i in xs %}{{ loop.index }}{% endfor %}{{ loop }}"
    assert _free_names(source) == ["xs", "loop"]


def test_a_tuple_target_binds_every_name_in_it():
    source = "{% for k, v in pairs %}{{ k }}{{ v }}{% endfor %}{{ v }}"
    assert _free_names(source) == ["pairs", "v"]


# ---------------------------------------------------------------------------
# `{% set %}` binds from where it appears
# ---------------------------------------------------------------------------

def test_set_binds_the_statements_after_it_and_not_before():
    """Jinja evaluates top to bottom, so `{{ x }}{% set x = 1 %}` really is an
    undefined reference followed by an assignment."""
    assert _free_names("{{ x }}{% set x = 1 %}{{ x }}") == ["x"]


def test_the_value_of_a_set_is_evaluated_before_the_name_exists():
    """In `{% set x = x %}` the right-hand `x` is whatever it meant before."""
    assert _free_names("{% set x = x %}{{ x }}") == ["x"]


def test_a_set_block_binds_after_the_block():
    assert _free_names("{% set x %}{{ y }}{% endset %}{{ x }}") == ["y"]


def test_a_set_inside_a_conditional_reaches_past_it():
    """`{% if %}` opens no scope in Jinja: a `{% set %}` in a taken branch is
    visible after the `{% endif %}`. Treating it as bound can only suppress a
    report, never invent one."""
    assert _free_names("{% if c %}{% set y = 1 %}{% endif %}{{ y }}") == ["c"]


def test_a_set_inside_a_loop_does_not_escape_it():
    source = "{% for i in xs %}{% set t = i %}{{ t }}{% endfor %}{{ t }}"
    assert _free_names(source) == ["xs", "t"]


# ---------------------------------------------------------------------------
# Macros, calls and with-blocks
# ---------------------------------------------------------------------------

def test_a_macro_argument_is_bound_only_in_its_body():
    source = "{% macro m(p) %}{{ p }}{% endmacro %}{{ p }}"
    assert _free_names(source) == ["p"]
    assert "p" in _refs(source)


def test_a_macro_name_is_callable_after_its_definition():
    assert _free_names("{% macro m(p) %}{{ p }}{% endmacro %}{{ m() }}") == []


def test_a_macro_default_is_evaluated_outside_the_body():
    assert _free_names("{% macro m(p=q) %}{{ p }}{% endmacro %}") == ["q"]


def test_caller_is_bound_inside_a_macro():
    """Jinja provides `caller` to a macro invoked through `{% call %}`."""
    assert _free_names("{% macro m() %}{{ caller() }}{% endmacro %}") == []


def test_a_with_block_binds_only_its_body():
    assert _free_names("{% with w = 1 %}{{ w }}{% endwith %}{{ w }}") == ["w"]


def test_a_call_block_argument_is_bound_only_in_its_body():
    source = "{% macro m() %}{{ caller(1) }}{% endmacro %}"
    source += "{% call(v) m() %}{{ v }}{% endcall %}{{ v }}"
    assert _free_names(source) == ["v"]


# ---------------------------------------------------------------------------
# The globals validator asks the same question
# ---------------------------------------------------------------------------

def test_a_global_shadowed_by_a_loop_is_left_alone_inside_it():
    """A template that binds `now` is talking about its own variable."""
    assert _misuse("{% for now in items %}{{ now }}{% endfor %}") == set()


@pytest.mark.parametrize(
    "source",
    [
        # Ours, before the loop that rebinds the name.
        "{{ now }}{% for now in items %}{{ now }}{% endfor %}",
        # Ours, after it.
        "{% for now in items %}{{ now }}{% endfor %}{{ now }}",
        # Ours, outside the macro that takes the name as an argument.
        "{% macro f(now) %}{{ now }}{% endmacro %}{{ now }}",
        # Ours, before the `{% set %}`.
        "{{ now }}{% set now = 'x' %}",
    ],
)
def test_a_global_outside_the_binding_is_still_ours(source):
    """The suppression must not extend past the binding.

    `{{ now }}` renders the repr of a live function object -- the defect #451
    exists to catch -- and a loop elsewhere in the file used to hide it.
    """
    assert NOT_CALLED in _misuse(source), (
        f"{source} names our global outside any binding of it and was accepted"
    )


def test_a_shadowed_and_an_unshadowed_use_can_share_one_template():
    """Identity, not spelling: same word, two nodes, two answers."""
    source = "{% for now in items %}{{ now }}{% endfor %}{{ now.foo }}"
    assert NOT_CALLED in _misuse(source)

"""A loop binds particular names, in particular places.

Validation treated "inside a loop" as one boolean and one union of names, and
both halves were wrong.

The boolean reached every field of a loop-shaped step, including the iterable
itself, so this was accepted::

    - id: process
      for_each: "{{ item.children }}"   # no item exists yet
      parameters:
        text: "{{ item.name }}"

The union meant every construct got every name, so ``{{ is_last }}`` inside a
`while` loop -- which binds no `is_last` -- passed validation and then failed
to render. That false negative arrived with #470, whose description claimed
the opposite; the correction is on #469.

`core.loop_contracts` holds one table per construct, read from the runtime
rather than from `docs/loop_variables.md`, which claims names the runtime does
not bind in every construct.
"""

import pytest

from orchestrator.core.loop_contracts import (
    ACTION_LOOP,
    ALL_BINDINGS,
    CREATE_PARALLEL_QUEUE,
    FOR_EACH,
    LOOP_CONTRACTS,
    WHILE,
    contract_for,
    is_source_field,
)
from orchestrator.validation.template_validator import TemplateValidator

pytestmark = [pytest.mark.contract]


def _errors(step, extra_context=None):
    result = TemplateValidator().validate_pipeline_templates(
        {"id": "p", "steps": [step]},
        extra_context or {},
    )
    return [(e.error_type, e.context_path) for e in result.errors]


# ---------------------------------------------------------------------------
# Source expressions are outside the loop they introduce
# ---------------------------------------------------------------------------

def test_a_loop_variable_in_the_iterable_is_rejected():
    """The reported case. The iterable must resolve before an item exists."""
    kinds = [k for k, _ in _errors(
        {"id": "s", "for_each": "{{ item.children }}", "parameters": {"t": "x"}}
    )]
    assert "loop_variable_outside_loop" in kinds, kinds


def test_the_error_points_at_the_iterable_not_the_body():
    paths = [p for k, p in _errors(
        {"id": "s", "for_each": "{{ item.children }}", "parameters": {"t": "x"}}
    ) if k == "loop_variable_outside_loop"]
    assert paths == ["steps[0].for_each"], paths


def test_the_same_name_is_accepted_in_the_body():
    """Source and body differ, so one template must not decide the other."""
    kinds = [k for k, _ in _errors(
        {"id": "s", "for_each": "{{ rows }}", "parameters": {"t": "{{ item.name }}"}},
        {"rows": [1]},
    )]
    assert kinds == [], kinds


@pytest.mark.parametrize("contract", sorted(set(LOOP_CONTRACTS.values()), key=lambda c: c.key))
def test_every_construct_declares_its_source_fields(contract):
    assert contract.source_fields, (
        f"{contract.key} declares no source field, so its iterable or "
        f"condition would be validated in loop scope"
    )
    for field in contract.source_fields:
        assert is_source_field(contract, field)


# ---------------------------------------------------------------------------
# Constructs bind different names
# ---------------------------------------------------------------------------

def test_a_name_another_construct_binds_is_rejected():
    """`while` has no `is_last`. Accepting the union is what let this pass."""
    kinds = [k for k, _ in _errors(
        {"id": "s", "while": "{{ go }}", "parameters": {"t": "{{ is_last }}"}},
        {"go": True},
    )]
    assert "loop_variable_wrong_construct" in kinds, kinds


def test_the_message_names_what_this_loop_does_bind():
    """A rejection that does not say what is available is a puzzle."""
    result = TemplateValidator().validate_pipeline_templates(
        {"id": "p", "steps": [
            {"id": "s", "while": "{{ go }}", "parameters": {"t": "{{ is_last }}"}}
        ]},
        {"go": True},
    )
    wrong = [e for e in result.errors if e.error_type == "loop_variable_wrong_construct"]
    assert wrong and "iteration" in wrong[0].suggestions[0], wrong[0].suggestions


@pytest.mark.parametrize("name", sorted(WHILE.bindings))
def test_while_accepts_what_while_binds(name):
    kinds = [k for k, _ in _errors(
        {"id": "s", "while": "{{ go }}", "parameters": {"t": "{{ %s }}" % name}},
        {"go": True},
    )]
    assert kinds == [], f"`while` binds {name} at run time but validation rejected it: {kinds}"


@pytest.mark.parametrize("name", sorted(FOR_EACH.bindings))
def test_for_each_accepts_what_for_each_binds(name):
    kinds = [k for k, _ in _errors(
        {"id": "s", "for_each": "{{ rows }}", "parameters": {"t": "{{ %s }}" % name}},
        {"rows": [1]},
    )]
    assert kinds == [], kinds


def test_a_queue_only_name_is_rejected_in_a_for_each():
    kinds = [k for k, _ in _errors(
        {"id": "s", "for_each": "{{ rows }}", "parameters": {"t": "{{ queue_size }}"}},
        {"rows": [1]},
    )]
    assert "loop_variable_wrong_construct" in kinds, kinds


def test_a_queue_only_name_is_accepted_in_a_queue():
    kinds = [k for k, _ in _errors(
        {
            "id": "s",
            "create_parallel_queue": {"on": "{{ rows }}"},
            "parameters": {"t": "{{ queue_size }}"},
        },
        {"rows": [1]},
    )]
    assert kinds == [], kinds


def test_an_action_loop_binds_no_item():
    """An action loop repeats actions; it is not iterating a collection."""
    assert "item" not in ACTION_LOOP.bindings
    kinds = [k for k, _ in _errors(
        {"id": "s", "action_loop": [{"action": "noop"}], "parameters": {"t": "{{ item }}"}}
    )]
    assert "loop_variable_wrong_construct" in kinds, kinds


# ---------------------------------------------------------------------------
# Scope still behaves
# ---------------------------------------------------------------------------

def test_a_loop_variable_outside_any_loop_is_still_rejected():
    kinds = [k for k, _ in _errors({"id": "s", "parameters": {"t": "{{ item }}"}})]
    assert "loop_variable_outside_loop" in kinds, kinds


def test_a_declared_input_named_item_still_wins():
    kinds = [k for k, _ in _errors(
        {"id": "s", "parameters": {"t": "{{ item }}"}}, {"item": "declared"}
    )]
    assert kinds == [], kinds


def test_a_nested_loop_sees_both_constructs():
    """An inner loop does not hide the outer one's bindings."""
    kinds = [k for k, _ in _errors(
        {
            "id": "outer",
            "for_each": "{{ rows }}",
            "steps": [{
                "id": "inner",
                "while": "{{ go }}",
                "parameters": {"t": "{{ item }} {{ iteration }}"},
            }],
        },
        {"rows": [1], "go": True},
    )]
    assert kinds == [], kinds


# ---------------------------------------------------------------------------
# One declaration
# ---------------------------------------------------------------------------

def test_the_union_is_built_from_the_contracts():
    from orchestrator.core.template_globals import ALL_LOOP_VARIABLES

    assert ALL_LOOP_VARIABLES is ALL_BINDINGS, (
        "the flat list must be a projection of the contracts, not a second "
        "copy that can drift"
    )


def test_every_loop_step_key_has_a_contract():
    from orchestrator.core.template_globals import LOOP_STEP_KEYS

    assert set(LOOP_STEP_KEYS) == set(LOOP_CONTRACTS), (
        "a key recognised as a loop without a contract would fall back to the "
        "union, which is the imprecision this replaces"
    )


def test_foreach_is_the_same_construct_as_for_each():
    """The compiler treats them identically, so validation must too."""
    assert LOOP_CONTRACTS["foreach"].bindings == FOR_EACH.bindings
    assert contract_for({"foreach": "x"}).bindings == FOR_EACH.bindings


def test_a_step_with_no_loop_key_has_no_contract():
    assert contract_for({"id": "s", "parameters": {}}) is None
    assert not is_source_field(None, "for_each")


@pytest.mark.parametrize("contract", sorted(set(LOOP_CONTRACTS.values()), key=lambda c: c.key))
def test_both_spellings_are_bound(contract):
    """The runtime registers `item` and `$item` alike."""
    for name in contract.bindings:
        assert f"${name}" in contract.all_bindings()


def test_the_parallel_queue_binds_more_than_the_others():
    """A guard on the tables being genuinely different -- if every construct
    ended up with the same set, the per-construct split would be decorative."""
    assert CREATE_PARALLEL_QUEUE.bindings - WHILE.bindings
    assert WHILE.bindings - CREATE_PARALLEL_QUEUE.bindings
    assert FOR_EACH.bindings != WHILE.bindings

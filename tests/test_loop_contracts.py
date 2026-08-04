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

#473 replaced the boolean with a flat tuple of "source fields" and made the
opposite error: it declared `action_loop` and `until` sources, which put the
*body* of an action loop outside the loop it is the body of, and it modelled
a `while` condition -- re-evaluated every iteration -- as if it resolved once
like a `for_each` iterable. Scope is now per field path.

These are validator tests: they check that the validator applies
`core.loop_contracts`. Whether those tables are *true* is a separate question,
and asking it here would be circular -- both sides would come from the same
table. `tests/test_loop_runtime_parity.py` answers it by executing pipelines.
"""

import pytest

from orchestrator.core.loop_contracts import (
    ACTION_LOOP,
    ALL_BINDINGS,
    CREATE_PARALLEL_QUEUE,
    FOR_EACH,
    LOOP_CONTRACTS,
    PRESTRIPPED_DOLLAR_NAMES,
    WHILE,
    contracts_for,
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
# A source expression is outside the loop it introduces
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


# ---------------------------------------------------------------------------
# ...but a condition is not a source: it is re-evaluated per iteration
# ---------------------------------------------------------------------------

def test_a_while_condition_reads_the_iteration_count():
    """`WhileLoopHandler.should_continue` assembles `iteration` before
    evaluating the condition, every iteration. #473 called `while` a source
    field and rejected the ordinary way of writing a bounded loop."""
    kinds = [k for k, _ in _errors(
        {"id": "s", "while": "{{ iteration < 3 }}", "steps": []}
    )]
    assert kinds == [], kinds


def test_a_while_condition_does_not_see_the_whole_body_scope():
    """It sees exactly what `should_continue` puts in scope -- not `index` or
    `position`, which the body has and the condition does not."""
    kinds = [k for k, _ in _errors(
        {"id": "s", "while": "{{ position > 1 }}", "steps": []}
    )]
    assert "loop_variable_wrong_construct" in kinds, kinds


def test_an_action_loop_body_is_inside_its_own_loop():
    """The `action_loop` key *is* the body. Declaring it a source field put
    iteration state out of reach of the only place it exists."""
    kinds = [k for k, _ in _errors({
        "id": "s",
        "action_loop": [{"action": "noop", "parameters": {"v": "{{ iteration }}"}}],
        "until": "{{ iteration >= 3 }}",
    })]
    assert kinds == [], kinds


def test_a_parallel_queue_source_is_outer_but_its_actions_are_not():
    """One object, two scopes: `on` builds the queue, the actions run per
    item. A whole-object source field cannot express that."""
    found = _errors({
        "id": "s",
        "create_parallel_queue": {
            "on": "{{ item }}",
            "action_loop": [{"action": "noop", "parameters": {"v": "{{ item }}"}}],
        },
    })
    assert found == [
        ("loop_variable_outside_loop", "steps[0].create_parallel_queue.on")
    ], found


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
# A step declares one loop
# ---------------------------------------------------------------------------

def test_a_step_declaring_two_loop_constructs_is_rejected():
    """Which one wins was decided by declaration order in `loop_contracts`,
    an order no engine agreed to, so the step's bindings depended on an
    implementation detail."""
    kinds = [k for k, _ in _errors(
        {"id": "s", "for_each": "{{ rows }}", "while": "{{ go }}", "parameters": {}},
        {"rows": [1], "go": True},
    )]
    assert "ambiguous_loop_construct" in kinds, kinds


def test_one_loop_construct_is_not_ambiguous():
    kinds = [k for k, _ in _errors(
        {"id": "s", "for_each": "{{ rows }}", "parameters": {}}, {"rows": [1]}
    )]
    assert "ambiguous_loop_construct" not in kinds, kinds


def test_contracts_for_reports_every_construct_a_step_declares():
    both = contracts_for({"for_each": "x", "while": "y"})
    assert {c.key for c in both} == {"for_each", "while"}
    assert contracts_for({"id": "s", "parameters": {}}) == ()
    assert contracts_for("not a step") == ()


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


def test_an_inner_loop_source_still_sees_the_outer_loop():
    """The inner iterable resolves inside the outer iteration, so the outer
    item is exactly what it is normally written from."""
    kinds = [k for k, _ in _errors(
        {
            "id": "outer",
            "for_each": "{{ rows }}",
            "steps": [{
                "id": "inner",
                "for_each": "{{ item.children }}",
                "parameters": {"t": "x"},
            }],
        },
        {"rows": [1]},
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


def test_foreach_is_not_a_loop_construct():
    """#470 and #473 accepted `foreach` as an alias of `for_each` on the
    strength of a comment. No engine expands it -- a `foreach` step fails
    schema validation outright (#475) -- so validating it as a working loop
    tells a reader the opposite of the truth."""
    assert "foreach" not in LOOP_CONTRACTS
    assert contracts_for({"foreach": "{{ rows }}"}) == ()


def test_the_dollar_spelling_is_an_observed_list_not_a_derived_one():
    """`{{ $position }}` is a compile error while `{{ $item }}` resolves,
    because the rewrite that makes `$` work does not reach every path (#474).
    Deriving the `$` set as "`$` + every binding" would accept the first."""
    assert "$item" in FOR_EACH.all_bindings()
    assert "$position" not in FOR_EACH.all_bindings()
    assert "position" in FOR_EACH.all_bindings()
    assert all(name.startswith("$") for name in PRESTRIPPED_DOLLAR_NAMES)


def test_a_narrowed_field_does_not_narrow_its_siblings():
    """`bindings_for` matches the longest prefix, so scoping
    `create_parallel_queue.on` leaves the object around it alone."""
    assert CREATE_PARALLEL_QUEUE.bindings_for("create_parallel_queue.on") == frozenset()
    assert "item" in CREATE_PARALLEL_QUEUE.bindings_for("create_parallel_queue.action_loop")
    assert "item" in CREATE_PARALLEL_QUEUE.bindings_for("parameters")


def test_the_tables_are_genuinely_different():
    """A guard on the per-construct split being real -- if every construct
    ended up with the same set, splitting them would be decorative."""
    assert CREATE_PARALLEL_QUEUE.bindings - WHILE.bindings
    assert WHILE.bindings - CREATE_PARALLEL_QUEUE.bindings
    assert FOR_EACH.bindings != WHILE.bindings
    assert "item" not in ACTION_LOOP.bindings

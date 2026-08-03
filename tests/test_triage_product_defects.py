"""Three defects the catalogue triage found wearing example failures as a disguise.

Classifying all 87 non-validating examples by their actual `validate` output
turned up three clusters that were not example problems at all. They are worth
a file together because they share a shape: each one reported *the pipeline* as
broken when the fault was ours, so anyone repairing the examples by hand would
have "fixed" files that were already correct.

1. A `TypeError` escaped the model validator and was reported as
   "Model validation failed", i.e. as though the pipeline were invalid.
2. `execution['timestamp']` was read as a task id, while `execution.timestamp`
   -- the same reference -- validated.
3. `json_encode` did not exist.
"""

import pytest

from orchestrator.core.template_manager import TemplateManager
from orchestrator.validation.data_flow_validator import DataFlowValidator
from orchestrator.validation.model_validator import ModelValidator

pytestmark = [pytest.mark.contract]


def _reference(ref, **kwargs):
    kwargs.setdefault("task_id", "some_task")
    kwargs.setdefault("parameter_name", "content")
    kwargs.setdefault("task_schemas", {})
    kwargs.setdefault("pipeline_inputs", {})
    return DataFlowValidator()._validate_variable_reference(ref, **kwargs)


# ---------------------------------------------------------------------------
# 1. A JSON Schema is not a model specification
# ---------------------------------------------------------------------------

#: The shape that crashed it: a schema whose `properties` names a field
#: called "name". Six catalogue examples carried one.
SCHEMA_PIPELINE = {
    "id": "p",
    "steps": [
        {
            "id": "validate_data",
            "action": "generate",
            "parameters": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "records": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "name": {"type": "string"},
                                    "active": {"type": "boolean"},
                                },
                            },
                        }
                    },
                }
            },
        }
    ],
}


def test_a_json_schema_is_not_read_as_a_model():
    """`properties.name` names a *field* called "name", not a model.

    Any nested dict carrying a `name` or `model` key used to be treated as a
    model specification. Here that made the "model name" a dict, and
    `validated_models.add(...)` raised `TypeError: unhashable type: 'dict'`.
    """
    result = ModelValidator().validate_pipeline_models(SCHEMA_PIPELINE)

    assert result.is_valid, (
        "a JSON Schema was read as a model specification: "
        f"{[e.message for e in result.errors]}"
    )
    assert not result.validated_models, (
        f"a schema field was recorded as a model: {result.validated_models}"
    )


def test_the_validator_does_not_raise_on_a_schema():
    """The crash itself, separately from the diagnosis it produced.

    The `TypeError` escaped and surfaced as "Model validation failed", so the
    report blamed the pipeline for a bug in the validator.
    """
    try:
        ModelValidator().validate_pipeline_models(SCHEMA_PIPELINE)
    except TypeError as exc:  # pragma: no cover - this is the regression
        pytest.fail(f"the model validator raised instead of reporting: {exc}")


def test_a_model_under_a_model_key_is_still_found():
    """Narrowing what counts as a model must not stop finding real ones."""
    pipeline = {
        "id": "p",
        "steps": [
            {
                "id": "think",
                "action": "generate",
                "parameters": {"model": "openai/gpt-4", "prompt": "hi"},
            }
        ],
    }
    result = ModelValidator().validate_pipeline_models(pipeline)
    assert "openai/gpt-4" in result.validated_models


def test_a_model_name_that_is_not_a_string_is_reported_not_raised():
    """Defence in depth for the same crash, at the other end.

    Scoping detection to model keys removes the way this was reached; a
    mapping written directly under `model:` must still be a message.
    """
    pipeline = {
        "id": "p",
        "steps": [
            {
                "id": "think",
                "action": "generate",
                "parameters": {"model": {"name": {"nested": "mapping"}}},
            }
        ],
    }
    result = ModelValidator().validate_pipeline_models(pipeline)
    assert not result.is_valid
    assert any("must be a string" in e.message for e in result.errors)


# ---------------------------------------------------------------------------
# 2. Two spellings of one reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dotted,subscript",
    [
        ("execution.timestamp", "execution['timestamp']"),
        ("execution.started_at", "execution['started_at']"),
        ('execution.id', 'execution["id"]'),
    ],
)
def test_both_spellings_of_a_reference_agree(dotted, subscript):
    """`a['b']` and `a.b` are the same reference and must be judged the same.

    Splitting on `.` left the subscript attached, so `execution['timestamp']`
    was looked up as a *task id* -- on pipelines that run correctly.
    """
    assert _reference(dotted)["valid"] == _reference(subscript)["valid"] is True


@pytest.mark.parametrize(
    "reference", ["execution['bogus']", 'execution["strated_at"]']
)
def test_a_typo_is_still_caught_in_the_subscript_spelling(reference):
    """Normalising the spelling must not smuggle unknown fields past the check."""
    result = _reference(reference)
    assert result["valid"] is False
    assert result["error_type"] == "unknown_execution_field"


def test_an_index_is_not_a_separate_name():
    """`records[0]` refers to `records`, not to something called `records[0]`."""
    assert _reference("records[0]")["valid"] == _reference("records")["valid"]


# ---------------------------------------------------------------------------
# 3. json_encode
# ---------------------------------------------------------------------------

def test_json_encode_is_the_same_function_as_to_json():
    """An alias, not a second implementation.

    Two implementations of one filter is exactly the drift #449 removed, and
    the conformance sweep would catch them diverging -- but sharing the object
    means they cannot.
    """
    filters = TemplateManager().env.filters
    assert filters["json_encode"] is filters["to_json"]


# ---------------------------------------------------------------------------
# 4. References were extracted by chopping text
# ---------------------------------------------------------------------------

def _refs(template):
    return DataFlowValidator()._extract_template_variables(template)


@pytest.mark.parametrize(
    "template,expected",
    [
        # An expression with a space in it. `split(' ')[0]` cut here.
        ("{{ zip(rows, cols) }}", {"zip", "rows", "cols"}),
        ("{{ 'yes' if ready else 'no' }}", {"ready"}),
        # A filter. `split('|')[0]` left the tail behind as its own "name".
        ("{{ payload | from_json }}", {"payload"}),
        # A subscript holding another reference.
        ("{{ topics[loop.index] }}", {"topics", "loop.index"}),
        # A chain is one reference, not three.
        ("{{ a.b.c }}", {"a.b.c"}),
    ],
)
def test_a_reference_survives_being_in_an_expression(template, expected):
    """Chopping the text produced fragments nobody wrote.

        Undefined task reference: '(row'
        Undefined task reference: 'from_json)'
        Undefined task reference: 'analysis_topics[loop'

    Each was then looked up as a task id and reported against the pipeline.
    """
    assert set(_refs(template)) == expected


def test_a_loop_target_is_not_a_reference():
    """`{% for row in rows %}` binds `row`; only `rows` is a reference."""
    assert set(_refs("{% for row in rows %}{{ row.x }}{% endfor %}")) == {"rows"}


@pytest.mark.parametrize("template", ["{{ range(3) }}", "{{ now() }}"])
def test_a_function_the_environment_provides_is_not_a_task(template):
    assert _refs(template) == []


def test_a_broken_template_yields_no_references():
    """Syntax is the template validator's business.

    Guessing at references inside something that does not parse is how the
    fragments above were invented.
    """
    assert _refs("{{ unclosed ") == []


@pytest.mark.parametrize("reference", ["loop.index", "item.name", "loop", "item"])
def test_loop_variables_are_recognised_however_they_are_written(reference):
    """The allowlist compared the *whole* reference, so a bare `loop` passed
    while `{{ loop.index }}` -- the way anyone writes it -- did not."""
    assert _reference(reference)["valid"] is True


def test_a_typo_inside_an_expression_is_now_caught():
    """Parsing catches more than chopping did, not less.

    `split(' ')[0]` on `'x' if ghost_step.done else 'y'` yielded `'x'` -- a
    literal, discarded -- so the reference to a step that does not exist was
    never checked at all. A 14-file jump in the catalogue is only trustworthy
    if it comes with this.
    """
    refs = _refs("{{ 'x' if ghost_step.done else 'y' }}")
    assert "ghost_step.done" in refs

    result = _reference("ghost_step.done")
    assert result["valid"] is False
    assert result["error_type"] == "undefined_task"

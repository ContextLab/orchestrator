"""A template in a field nothing renders is not a runtime promise.

`core.step_fields` established which parts of a step the runtime renders, and
#471 wired it into dependency inference so two inert strings stopped inventing
a cycle. Template validation was still reading those same fields as if they
resolved, and got both directions wrong:

    - id: a
      name: "{{ b.result }}"          # "will be resolved at runtime" -- it is not
      description: "{{ nosuch }}"     # a hard error, in prose nobody renders

The first is a false promise: nothing substitutes into `name`, so the braces
reach the log verbatim and the reader is told the opposite. The second is a
false rejection: a stray brace in a description failed a pipeline that runs
correctly, which is the class of bug #465, #469 and #472 each removed
elsewhere.

The run in `test_a_pipeline_with_templates_in_prose_still_runs` is the check
that this is a real property of the runtime and not a claim about it.
"""

import asyncio
from pathlib import Path

import pytest

from orchestrator.core.step_fields import (
    INERT_STEP_FIELDS,
    RENDERABLE_STEP_FIELDS,
)
from orchestrator.validation.template_validator import TemplateValidator
from tests.test_infrastructure import create_test_orchestrator

pytestmark = [pytest.mark.contract]

#: `id` names the step and `dependencies`/`depends_on` hold step ids. They are
#: inert for the same reason, but a template in one is a structural error
#: rather than prose, so they are not probed as free text here.
PROSE_FIELDS = sorted(INERT_STEP_FIELDS - {"id", "dependencies", "depends_on", "tool"})


def _validate(step, context=None):
    return TemplateValidator().validate_pipeline_templates(
        {"id": "p", "steps": [step, {"id": "b", "parameters": {}}]},
        context or {},
    )


@pytest.mark.parametrize("field", PROSE_FIELDS)
def test_an_undefined_name_in_an_inert_field_is_not_an_error(field):
    """It cannot be undefined: it is never looked up."""
    result = _validate({"id": "a", field: "{{ nosuch_variable }}"})
    assert result.is_valid, [
        (e.error_type, e.context_path, e.message) for e in result.errors
    ]


@pytest.mark.parametrize("field", PROSE_FIELDS)
def test_an_inert_field_still_warns(field):
    """Silence would be wrong too -- the author wrote a template and will get
    braces. The warning is how they find out before reading the output."""
    result = _validate({"id": "a", field: "{{ b.result }}"})
    kinds = [w.error_type for w in result.warnings]
    assert "inert_field_template" in kinds, kinds


def test_the_warning_does_not_claim_the_value_arrives_later():
    """The old message said `will be resolved at runtime`, which is the one
    thing that does not happen."""
    result = _validate({"id": "a", "name": "{{ b.result }}"})
    inert = [w for w in result.warnings if w.error_type == "inert_field_template"]
    assert inert, [w.error_type for w in result.warnings]
    assert "resolved at runtime" not in inert[0].message
    assert "never rendered" in inert[0].message


def test_the_warning_names_the_inert_field_not_the_key_beneath_it():
    """`metadata.note` is inert because `metadata` is. Naming `note` would
    send the reader looking for a rule about a key they invented."""
    result = _validate({"id": "a", "metadata": {"note": "{{ b.result }}"}})
    inert = [w for w in result.warnings if w.error_type == "inert_field_template"]
    assert inert and "'metadata'" in inert[0].message, [w.message for w in inert]
    assert inert[0].context_path == "steps[0].metadata.note", inert[0].context_path


@pytest.mark.parametrize("field", RENDERABLE_STEP_FIELDS)
def test_a_renderable_field_still_reports_an_undefined_name(field):
    """The suppression must not spread. These fields do resolve, so a name
    that is not there is still an error."""
    value = {"x": "{{ nosuch_variable }}"} if field == "parameters" else "{{ nosuch_variable }}"
    result = _validate({"id": "a", field: value})
    assert not result.is_valid, f"{field} is rendered; an undefined name there is an error"
    assert "undefined_variable" in [e.error_type for e in result.errors]


def test_a_step_result_reference_in_a_rendered_field_is_still_a_runtime_promise():
    result = _validate({"id": "a", "parameters": {"x": "{{ b.result }}"}})
    assert "runtime_variable" in [w.error_type for w in result.warnings]


@pytest.mark.e2e
def test_a_pipeline_with_templates_in_prose_still_runs(tmp_path):
    """The evidence that these fields are inert, rather than the assertion.

    Every prose field carries a reference to a name that exists nowhere. If
    any of them were rendered the run would fail or write the wrong thing; the
    file that lands proves it did neither.
    """
    pipeline = f"""
id: inert_prose
name: "{{{{ nosuch_pipeline_name }}}}"
description: "{{{{ nosuch_description }}}}"
metadata:
  owner: "{{{{ nosuch_owner }}}}"
steps:
  - id: write_it
    name: "{{{{ nosuch_step_name }}}}"
    description: "{{{{ nosuch_step_description }}}}"
    metadata:
      note: "{{{{ nosuch_metadata }}}}"
    tool: filesystem
    action: write
    parameters:
      path: "{tmp_path}/out.txt"
      content: "written"
"""
    asyncio.run(create_test_orchestrator().execute_yaml(pipeline, {}))
    written = Path(tmp_path, "out.txt")
    assert written.exists(), "the step did not run"
    assert written.read_text() == "written"


@pytest.mark.e2e
def test_that_same_pipeline_validates(tmp_path):
    """Both surfaces agree: what runs, validates."""
    result = TemplateValidator().validate_pipeline_templates(
        {
            "id": "inert_prose",
            "name": "{{ nosuch_pipeline_name }}",
            "steps": [{
                "id": "write_it",
                "name": "{{ nosuch_step_name }}",
                "description": "{{ nosuch_step_description }}",
                "metadata": {"note": "{{ nosuch_metadata }}"},
                "tool": "filesystem",
                "action": "write",
                "parameters": {"path": str(tmp_path / "out.txt"), "content": "written"},
            }],
        },
        {},
    )
    assert result.is_valid, [(e.error_type, e.context_path) for e in result.errors]

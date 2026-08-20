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
    INERT_PROSE_STEP_FIELDS,
    NON_RENDERED_STRUCTURAL_STEP_FIELDS,
    OPERATIONAL_METADATA_KEYS,
    RENDERABLE_STEP_FIELDS,
)
from orchestrator.validation.template_validator import TemplateValidator
from tests.test_infrastructure import create_test_orchestrator

pytestmark = [pytest.mark.contract]

#: Fields that are prose: unrendered, and harmless. `metadata` joins them
#: because arbitrary author data is prose too -- its *reserved* keys are not,
#: and those are covered separately below.
PROSE_FIELDS = sorted(INERT_PROSE_STEP_FIELDS | {"metadata"})


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


# ---------------------------------------------------------------------------
# Unrendered does not mean harmless
# ---------------------------------------------------------------------------
#
# The first version of this module treated every unrendered field as prose and
# warned. That said a `goto` sending execution to a step literally named
# `{{ nosuch }}` was a wording problem, and reported a pipeline carrying it as
# valid.


@pytest.mark.parametrize("field", sorted(NON_RENDERED_STRUCTURAL_STEP_FIELDS))
def test_a_template_in_a_structural_field_is_an_error(field):
    """These fields *name* things -- a step, a tool, a dependency. A literal
    `{{ x }}` names nothing, so the pipeline is already broken."""
    value = ["{{ nosuch }}"] if field in ("dependencies", "depends_on") else "{{ nosuch }}"
    result = _validate({"id": "a", field: value})
    assert not result.is_valid, f"a template in '{field}' cannot resolve to a name"
    assert "template_in_structural_field" in [e.error_type for e in result.errors]


@pytest.mark.parametrize("key", sorted(OPERATIONAL_METADATA_KEYS))
def test_a_template_in_runtime_read_metadata_is_an_error(key):
    """Each of these keys has a runtime read behind it, so the literal
    template text is what control code would act on."""
    result = _validate({"id": "a", "metadata": {key: "{{ nosuch }}"}})
    assert not result.is_valid, f"metadata '{key}' is read by the runtime"
    assert "template_in_operational_metadata" in [e.error_type for e in result.errors]


def test_arbitrary_metadata_is_still_prose():
    """The split must not swallow the case it started from: a note an author
    wrote for themselves is not a defect."""
    result = _validate({"id": "a", "metadata": {"note": "{{ nosuch }}"}})
    assert result.is_valid, [(e.error_type, e.message) for e in result.errors]
    assert "inert_field_template" in [w.error_type for w in result.warnings]


def test_an_operational_key_is_only_operational_inside_metadata():
    """`priority` nested deeper is somebody's data structure, not the key the
    runtime reads."""
    result = _validate({"id": "a", "metadata": {"notes": {"priority": "{{ nosuch }}"}}})
    assert result.is_valid, [(e.error_type, e.message) for e in result.errors]


def test_metadata_is_not_traversed_as_pipeline_structure():
    """`metadata` holds arbitrary author data, so a key named `steps` inside it
    is data. It was being walked as pipeline structure, and a dict carrying
    `for_each` and `while` reported an ambiguous loop -- from inside a subtree
    this module had just declared the runtime copies verbatim."""
    result = _validate({"id": "a", "metadata": {"steps": [{"for_each": "x", "while": "y"}]}})
    assert "ambiguous_loop_construct" not in [e.error_type for e in result.errors]
    assert result.is_valid, [(e.error_type, e.context_path) for e in result.errors]


def test_the_three_classes_are_disjoint():
    """A field in two of them would be reported by whichever check ran first."""
    assert not (INERT_PROSE_STEP_FIELDS & NON_RENDERED_STRUCTURAL_STEP_FIELDS)
    assert not (set(RENDERABLE_STEP_FIELDS) & INERT_PROSE_STEP_FIELDS)
    assert not (set(RENDERABLE_STEP_FIELDS) & NON_RENDERED_STRUCTURAL_STEP_FIELDS)


@pytest.mark.e2e
def test_the_specific_diagnostic_code_survives_to_the_api(tmp_path):
    """`create_template_issue` hardcoded `template_error`, so every template
    finding arrived as the same code and a consumer had to read the prose to
    tell an inert-field note from a loop-scope error."""
    from orchestrator.validation.pipeline_report import validate_pipeline_file

    document = tmp_path / "p.yaml"
    document.write_text(
        "id: codes\nname: codes\nsteps:\n"
        '  - id: a\n    name: "{{ nosuch }}"\n    tool: filesystem\n'
        "    action: write\n    parameters:\n"
        f'      path: "{tmp_path}/x.txt"\n      content: hi\n'
    )
    result = validate_pipeline_file(document)
    assert result.valid, result.error
    assert "inert_field_template" in [f.code for f in result.findings], [
        f.code for f in result.findings
    ]

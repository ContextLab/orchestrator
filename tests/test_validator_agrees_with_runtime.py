"""`validate` must not reject what `run` executes correctly.

`orchestrator validate` rejected 108 of 117 example pipelines. Only 23 of them
actually fail to compile when run; the other 85 execute fine. The gate was
measuring the validator's strictness rather than whether a pipeline works, and
every one of those false rejections told an author their working pipeline was
broken.

Three disagreements produced almost all of it:

1. The validator's Jinja environment was missing 14 filters the runtime
   registers, so `{{ title | slugify }}` was "Unknown filter: 'slugify'".
2. Neither validator recognised the `parameters.` / `inputs.` namespaces. The
   data-flow validator went further and read `parameters` as a *task id*.
3. A task's available outputs came from a hardcoded list of twelve field names,
   so any domain-specific field was "does not produce output".

The rule these tests pin is narrow and checkable: **anything the runtime
renders, the validator must accept.** The reverse is not required -- the
validator may still catch things the runtime would fail on later.
"""

import pytest

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.core.template_manager import TemplateManager
from orchestrator.validation.template_validator import TemplateValidator

pytestmark = [pytest.mark.contract]


def _validate(yaml_text):
    """Compile the way `orchestrator validate` does. Returns None or the error."""
    import asyncio

    try:
        asyncio.run(YAMLCompiler().compile(yaml_text, {}))
        return None
    except Exception as exc:  # noqa: BLE001 - the message is the subject
        return f"{type(exc).__name__}: {exc}"


def _pipeline(content, extra_params=""):
    return f"""
id: probe
name: Probe
parameters:
  topic:
    type: string
    default: "Hello World Report"
{extra_params}
steps:
  - id: write_it
    tool: filesystem
    action: write
    parameters:
      path: "./out.txt"
      content: "{content}"
"""


# ---------------------------------------------------------------------------
# 1a. Filters
# ---------------------------------------------------------------------------

def test_every_environment_knows_every_filter_the_runtime_registers():
    """The drift check.

    Three environments built their own filter sets: the validator knew 56, the
    compiler 61, the runtime 70. Which error a pipeline got depended on which
    environment reached it first -- `{{ title | slugify }}` failed validation,
    and fixing only the validator moved the failure to `truncate_words` in the
    compiler.
    """
    runtime = set(TemplateManager().env.filters)
    others = {
        "template validator": set(TemplateValidator().env.filters),
        "yaml compiler": set(YAMLCompiler().template_engine.filters),
    }

    drifted = {
        name: sorted(runtime - filters)
        for name, filters in others.items()
        if runtime - filters
    }
    assert not drifted, (
        f"environments are missing filters the runtime registers, so pipelines "
        f"using them are rejected despite working: {drifted}"
    )


@pytest.mark.parametrize(
    "expression",
    [
        "{{ topic | slugify }}",
        "{{ '/a/b/c.txt' | basename }}",
        "{{ topic | regex_search('World') }}",
        "{{ topic | truncate_words(2) }}",
        # Valid JSON, because the subject is whether the filter is *known*.
        # This case used to feed `from_json` the literal "Hello World Report",
        # which is not JSON, so the pipeline failed for an unrelated reason and
        # the assertion below was satisfied by the wrong error.
        "{{ '[1, 2]' | from_json }}",
    ],
)
def test_a_pipeline_using_a_runtime_filter_validates(expression):
    # Assert it compiles. The previous form accepted any error whose text did
    # not contain "filter", so an unrelated rendering failure passed as though
    # the filter had been recognised.
    assert _validate(_pipeline(expression)) is None, (
        f"{expression} does not validate, though the runtime registers this filter"
    )


# ---------------------------------------------------------------------------
# 1b. Parameter namespaces
# ---------------------------------------------------------------------------

def test_a_parameter_without_a_default_is_still_declared():
    """Having no default does not make a parameter undeclared.

    Its value arrives at run time from `-i name=value`. The validator used to
    register only parameters that carried a default, so a pipeline declaring
    `output_path` was told `Undefined variable: 'output_path'` -- the name it
    had just declared. This was the largest single cluster of false
    rejections in the catalogue.
    """
    pipeline = """
id: no_default
name: No Default
parameters:
  output_path:
    type: string
    description: "where to write"
steps:
  - id: s
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}"
      content: "x"
"""
    assert _validate(pipeline) is None, (
        "a declared parameter without a default was reported undefined"
    )


@pytest.mark.parametrize(
    "expression",
    ["{{ topic }}", "{{ parameters.topic }}", "{{ inputs.topic }}"],
)
def test_every_way_of_naming_a_parameter_validates(expression):
    """All three render identically at runtime; all three must validate.

    Before this, only the bare form did.
    """
    assert _validate(_pipeline(expression)) is None, (
        f"{expression} does not validate, but running it renders the parameter "
        f"correctly"
    )


# ---------------------------------------------------------------------------
# 1c. Step output fields
# ---------------------------------------------------------------------------

def test_a_step_may_produce_a_field_the_validator_cannot_know():
    """A model or AUTO step's output shape is not statically knowable.

    Rejecting an unrecognised field asserts knowledge the validator does not
    have. It may warn; it must not refuse to compile.
    """
    pipeline = """
id: custom_fields
name: Custom Fields
steps:
  - id: analyse
    action: <AUTO>classify the input and return a strategy</AUTO>

  - id: use_it
    tool: filesystem
    action: write
    parameters:
      path: "./out.txt"
      content: "{{ analyse.strategy }} / {{ analyse.confidence }}"
    dependencies:
      - analyse
"""
    # Assert it compiles, not that some particular wording is absent. Checking
    # for the old message would pass the moment the message was reworded, even
    # with the rejection still in force -- which is exactly what happened when
    # this was first written.
    assert _validate(pipeline) is None, (
        "the validator refused a pipeline over output fields it cannot know"
    )

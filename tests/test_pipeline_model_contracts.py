"""Contract tests for pipelines that use a model.

The golden pipelines (tests/test_golden_pipelines.py) cover the tool-only
path: no model, no provider. This module covers the other half -- a pipeline
whose steps call a model -- using the deterministic model and provider from
tests/test_infrastructure.py, so it stays hermetic: no network, no API key, no
provider SDK.

These replace src/orchestrator/testing/pipeline_integration_infrastructure.py,
retired in #435. That module built a parallel testing architecture -- its own
model, provider, validator, scoring system and orchestration facade -- inside
the shipped package, was written against constructor arguments that do not
exist, and consequently never ran. What was worth keeping was the *intent*:
prove that a model-using pipeline compiles, executes, fails honestly, and can
produce structured output. That intent is these tests, against the canonical
YAMLCompiler and Orchestrator and nothing else.

Test-only models and providers stay under tests/. They are not product code.
"""

import asyncio
import json
from collections.abc import Mapping

import jsonschema
import pytest

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.core.exceptions import YAMLCompilerError
from orchestrator.models.model_registry import ModelRegistry
from tests.test_infrastructure import MockTestModel, create_test_orchestrator

pytestmark = [pytest.mark.contract, pytest.mark.e2e]

# Jinja delimiters that must never survive into a result. A step that emits
# "{{ topic }}" has silently shipped its own source instead of a value (#153).
UNRESOLVED_TEMPLATE_MARKERS = ("{{", "}}", "{%", "%}")


def assert_no_unresolved_templates(text: str) -> None:
    for marker in UNRESOLVED_TEMPLATE_MARKERS:
        assert marker not in text, (
            f"unresolved template marker {marker!r} survived into the result: {text!r}"
        )


TOOL_PIPELINE = """
id: canonical_tool_pipeline
name: Canonical Tool Pipeline
description: A deterministic local-tool step
steps:
  - id: write_note
    tool: filesystem
    action: write
    parameters:
      path: "out/note.txt"
      content: "hello"
"""

# `{{ topic }}` is deliberate: it makes the "no unresolved templates" assertion
# meaningful. A pipeline with no templates would pass that check vacuously.
MODEL_TOPIC = "the water cycle"

MODEL_PIPELINE = """
id: canonical_model_pipeline
name: Canonical Model Pipeline
description: A single generate step against the deterministic test model
parameters:
  topic:
    type: string
    default: "unsubstituted-default"
steps:
  - id: think
    action: generate
    parameters:
      prompt: "Summarise the following: {{ topic }}"
"""

# `steps` must be a list of steps. A mapping where a list belongs is malformed.
MALFORMED_PIPELINE = """
id: broken_pipeline
name: Broken Pipeline
steps:
  not_a_list: true
"""

# One source of truth: the pipeline declares this schema and the test validates
# against this same object, so the two cannot drift apart. YAML is a superset of
# JSON, so the dumped schema drops straight into the document.
STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {"test_output": {"type": "string"}},
    "required": ["test_output"],
    "additionalProperties": False,
}


def structured_pipeline(action: str) -> str:
    return f"""
id: canonical_structured_pipeline
name: Canonical Structured Pipeline
steps:
  - id: extract
    action: {action}
    parameters:
      prompt: "Extract the fields"
      schema: {json.dumps(STRUCTURED_SCHEMA)}
"""

FAILING_PIPELINE = """
id: failing_pipeline
name: Failing Pipeline
steps:
  - id: read_missing
    action: file
    parameters:
      action: read
      path: "/nonexistent/definitely/not/here.txt"
"""


@pytest.fixture
def compiler():
    """A real YAMLCompiler backed by the deterministic test model."""
    registry = ModelRegistry()
    registry.register_model(MockTestModel())
    return YAMLCompiler(model_registry=registry)


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

def test_a_valid_pipeline_compiles(compiler):
    """The canonical compiler accepts a well-formed pipeline."""
    pipeline = asyncio.run(
        compiler.compile(TOOL_PIPELINE, {}, resolve_ambiguities=False)
    )

    assert pipeline.id == "canonical_tool_pipeline"
    assert "write_note" in pipeline.tasks
    assert pipeline.tasks["write_note"].action == "write"


def test_a_malformed_pipeline_is_refused(compiler):
    """Validation failure must raise rather than compile to something broken."""
    with pytest.raises(YAMLCompilerError):
        asyncio.run(
            compiler.compile(MALFORMED_PIPELINE, {}, resolve_ambiguities=False)
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#241 / #104: a model step runs but does not validate. Strict "
        "validation resolves `action: generate` as a TOOL name and reports "
        "\"Tool 'generate' not found in registry\", while the executor runs "
        "the same pipeline happily -- see test_a_model_pipeline_executes "
        "below. strict=True so this flips to a visible failure the moment "
        "the two agree, rather than sitting here forgotten."
    ),
)
def test_a_model_pipeline_compiles(compiler):
    """`validate` and `run` must accept the same pipelines. Today they do not."""
    pipeline = asyncio.run(
        compiler.compile(MODEL_PIPELINE, {}, resolve_ambiguities=False)
    )

    assert "think" in pipeline.tasks


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------

def test_a_model_pipeline_executes():
    """The orchestrator selects the deterministic model and runs the step.

    Asserting only that the step key exists would pass for a step that failed,
    returned nothing, or shipped its own unrendered template. Each of those has
    happened here, so each is pinned.
    """
    orchestrator = create_test_orchestrator()
    result = asyncio.run(
        orchestrator.execute_yaml(
            yaml_content=MODEL_PIPELINE, context={"topic": MODEL_TOPIC}
        )
    )

    assert "think" in result, f"step output missing from {sorted(result)}"
    step = result["think"]
    assert isinstance(step, Mapping), f"expected a result envelope, got {type(step)}"

    # the envelope
    assert step["success"] is True, f"step did not succeed: {step}"
    assert step.get("error") in (None, ""), f"a successful step must carry no error: {step}"
    assert step["action"] == "generate_text"

    # the model that was actually selected, by identity
    assert step["model_used"] == MockTestModel().name

    # the response
    response = step["result"]
    assert isinstance(response, str), f"expected text, got {type(response)}"
    assert response.strip(), "a successful generate step must return a non-empty response"

    # the template was rendered before the model saw it
    assert MODEL_TOPIC in response, (
        f"the rendered parameter never reached the model: {response!r}"
    )
    assert "unsubstituted-default" not in response
    assert_no_unresolved_templates(response)


@pytest.mark.parametrize("action", ["generate-structured", "generate_structured"])
def test_structured_output_pipeline_returns_an_object(action):
    """A structured step returns a schema-conforming object, not a sentence.

    Both spellings are pinned. Every other action in the project uses
    underscores, so `generate_structured` reads as the canonical name -- but
    only `generate-structured` used to dispatch. The underscore spelling fell
    through to the natural-language branch, which turns an unrecognised action
    into a prompt: the step returned a *string* and still reported success.
    """
    orchestrator = create_test_orchestrator()
    result = asyncio.run(
        orchestrator.execute_yaml(yaml_content=structured_pipeline(action), context={})
    )

    assert "extract" in result, f"step output missing from {sorted(result)}"
    payload = result["extract"]

    assert isinstance(payload, Mapping), (
        f"a structured step must return an object, got {type(payload)}: {payload!r}"
    )
    # Conformance, not merely dict-ness.
    jsonschema.validate(instance=payload, schema=STRUCTURED_SCHEMA)
    assert_no_unresolved_templates(json.dumps(payload))


def test_execution_failure_is_reported_not_swallowed():
    """A step that cannot succeed must surface as a failure.

    A pipeline reporting success having done nothing is the failure mode this
    project has repeatedly had to fix, so it is pinned here.
    """
    orchestrator = create_test_orchestrator()
    result = asyncio.run(
        orchestrator.execute_yaml(yaml_content=FAILING_PIPELINE, context={})
    )

    step = result["read_missing"]
    assert step["success"] is False, "reading a missing file must not report success"
    assert step["error"], "a failed step must say why"

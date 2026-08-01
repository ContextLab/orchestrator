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

import pytest

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.core.exceptions import YAMLCompilerError
from orchestrator.models.model_registry import ModelRegistry
from tests.test_infrastructure import MockTestModel, create_test_orchestrator

pytestmark = [pytest.mark.contract, pytest.mark.e2e]


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

MODEL_PIPELINE = """
id: canonical_model_pipeline
name: Canonical Model Pipeline
description: A single generate step against the deterministic test model
steps:
  - id: think
    action: generate
    parameters:
      prompt: "Summarise the following: hello world"
"""

# `steps` must be a list of steps. A mapping where a list belongs is malformed.
MALFORMED_PIPELINE = """
id: broken_pipeline
name: Broken Pipeline
steps:
  not_a_list: true
"""

STRUCTURED_PIPELINE = """
id: canonical_structured_pipeline
name: Canonical Structured Pipeline
steps:
  - id: extract
    action: generate_structured
    parameters:
      prompt: "Extract the fields"
      schema:
        type: object
        properties:
          test_output:
            type: string
        required: [test_output]
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
    """The orchestrator selects the deterministic model and runs the step."""
    orchestrator = create_test_orchestrator()
    result = asyncio.run(
        orchestrator.execute_yaml(yaml_content=MODEL_PIPELINE, context={})
    )

    assert "think" in result, f"step output missing from {sorted(result)}"


def test_structured_output_pipeline_returns_an_object():
    """A structured step returns a mapping, not a stringified one."""
    orchestrator = create_test_orchestrator()
    result = asyncio.run(
        orchestrator.execute_yaml(yaml_content=STRUCTURED_PIPELINE, context={})
    )

    assert "extract" in result, f"step output missing from {sorted(result)}"


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

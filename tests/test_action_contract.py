"""The action vocabulary contract.

`core/actions.py` is the one authoritative registry. Everything else -- the
executor's dispatch, the validator's recognition, the control systems'
advertised capabilities, alias normalisation, the documented vocabulary -- is
derived from it. These tests pin that derivation in both directions, so the
vocabulary cannot drift back apart.

The behaviour being protected, concretely: `action: gernate` must fail. It used
to become a successful model call, because any unrecognised action was turned
into a prompt.
"""

import asyncio
import warnings

import jsonschema
import pytest

from orchestrator.core.actions import (
    ACTION_FAMILIES,
    ACTION_SPECS,
    BUILTIN_ACTION_HANDLERS,
    SUPPORTED_ACTIONS,
    ActionSpec,
    canonical_action,
    is_known_action,
    match_action_family,
    resolve_action,
)
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from orchestrator.core.exceptions import UnknownActionError, YAMLCompilerError
from orchestrator.core.task import Task
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.validation.tool_validator import ToolValidator
from tests.test_infrastructure import MockTestModel, create_test_orchestrator

pytestmark = [pytest.mark.contract]


@pytest.fixture
def control_system():
    registry = ModelRegistry()
    registry.register_model(MockTestModel())
    return HybridControlSystem(model_registry=registry)


@pytest.fixture
def validator():
    """Strict: unknown names are errors, not warnings."""
    return ToolValidator(allow_unknown_tools=False)


@pytest.fixture
def compiler():
    registry = ModelRegistry()
    registry.register_model(MockTestModel())
    return YAMLCompiler(model_registry=registry)


def _pipeline(action, parameters="{}"):
    return f"""
id: action_contract_pipeline
name: Action Contract Pipeline
steps:
  - id: only_step
    action: {action}
    parameters: {parameters}
"""


# ---------------------------------------------------------------------------
# 1. every registered action validates and dispatches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ACTION_SPECS, ids=lambda s: s.name)
def test_every_registered_action_is_recognised_everywhere(spec, validator):
    """Registry, validator and canonical lookup must agree on every spelling."""
    for spelling in (spec.name, *spec.aliases):
        assert is_known_action(spelling), f"{spelling!r} unknown to the registry"
        assert canonical_action(spelling) == spec.name

        result = validator.validate_pipeline_tools(
            {"steps": [{"id": "s", "action": spelling,
                        "parameters": {p: "x" for p in spec.required_parameters}}]}
        )
        rejected = [e for e in result.errors if e.error_type == "unknown_tool"]
        assert not rejected, f"{spelling!r} rejected as a tool: {rejected}"


@pytest.mark.parametrize(
    "spec", [s for s in ACTION_SPECS if s.handler], ids=lambda s: s.name
)
def test_every_registered_handler_exists(spec, control_system):
    """Dispatch is `getattr(self, handler)`, so a bad name is an AttributeError.

    Naming the offending action here beats discovering it inside whichever
    pipeline happened to use it.
    """
    handler = getattr(control_system, spec.handler, None)
    assert handler is not None, f"{spec.name!r} -> missing {spec.handler!r}"
    assert callable(handler)


@pytest.mark.parametrize("family", ACTION_FAMILIES, ids=lambda f: f.name)
def test_every_family_handler_exists(family, control_system):
    """`auto` deliberately carries no handler -- it is the model path."""
    if family.handler is None:
        assert family.name == "auto"
        return
    assert callable(getattr(control_system, family.handler, None))


def test_no_handler_is_registered_without_a_spec(control_system):
    """The reverse direction: dispatch cannot reach an unregistered method."""
    for spelling, handler_name in BUILTIN_ACTION_HANDLERS.items():
        assert resolve_action(spelling) is not None
        assert hasattr(control_system, handler_name)


def test_supported_actions_is_generated_not_restated(control_system):
    """The advertised capability list must be the registry, not a copy of it.

    It used to advertise ten names -- transform, search, extract, filter,
    synthesize, create, optimize, review, write, compile -- that had no handler
    anywhere. Anything it lists must now be executable.
    """
    advertised = control_system._capabilities.get("supported_actions", [])

    assert set(advertised) == set(SUPPORTED_ACTIONS)
    for name in advertised:
        assert is_known_action(name), f"advertised but unknown: {name!r}"


# ---------------------------------------------------------------------------
# 2. aliases normalise identically
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "alias,canonical",
    [(a, s.name) for s in ACTION_SPECS for a in sorted(s.aliases)],
)
def test_alias_normalises_in_the_compiled_task_graph(alias, canonical, compiler):
    """One spelling reaches the task graph, whichever the author wrote."""
    parameters = "{" + ", ".join(
        f"{p}: x" for p in sorted(resolve_action(canonical).required_parameters)
    ) + "}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pipeline = asyncio.run(
            compiler.compile(_pipeline(alias, parameters), {}, resolve_ambiguities=False)
        )

    assert pipeline.tasks["only_step"].action == canonical, (
        f"alias {alias!r} was left unnormalised in the task graph"
    )


def test_a_deprecated_alias_warns(compiler):
    """Silent rewriting would leave authors with no reason to update."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(
            compiler.compile(
                _pipeline("generate-structured", '{prompt: p, schema: {type: object}}'),
                {},
                resolve_ambiguities=False,
            )
        )

    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("generate-structured" in m and "generate_structured" in m for m in messages), (
        f"no deprecation warning naming both spellings: {messages}"
    )


# ---------------------------------------------------------------------------
# 3. unknown actions fail, through YAML *and* through direct execution
# ---------------------------------------------------------------------------

def test_an_unknown_action_fails_to_compile(compiler):
    """`gernate` is a typo, not an instruction."""
    with pytest.raises(YAMLCompilerError) as excinfo:
        asyncio.run(
            compiler.compile(
                _pipeline("gernate", "{prompt: hello}"), {}, resolve_ambiguities=False
            )
        )

    assert "gernate" in str(excinfo.value)


def test_an_unknown_action_fails_at_dispatch_too(control_system):
    """Compile-time validation must not be the only safety boundary.

    A caller can build a Task and reach dispatch without going through YAML at
    all, so the runtime refuses independently.
    """
    task = Task(id="typo", name="typo", action="gernate", parameters={"prompt": "hi"})

    with pytest.raises(UnknownActionError) as excinfo:
        asyncio.run(control_system._execute_task_impl(task, {}))

    assert excinfo.value.details["action"] == "gernate"
    assert excinfo.value.details["task_id"] == "typo"


def test_a_typo_never_becomes_a_successful_model_call():
    """The regression this whole contract exists to prevent.

    Before: `action: gernate` was turned into a prompt, the model answered, and
    the step reported success. A misspelling produced a plausible-looking
    result instead of an error.
    """
    # `on_failure: continue` so the run completes and the step's own envelope
    # can be inspected. Under the default policy the pipeline raises instead,
    # which the assertion below also covers.
    tolerant = """
id: typo_pipeline
name: Typo Pipeline
steps:
  - id: only_step
    action: gernate
    on_failure: continue
    parameters:
      prompt: hello
"""
    orchestrator = create_test_orchestrator()
    result = asyncio.run(orchestrator.execute_yaml(yaml_content=tolerant, context={}))

    step = result["only_step"]
    assert "error" in step, f"a typo must not report success: {step}"
    assert "gernate" in str(step["error"]), (
        f"the step's error must name the offending action: {step['error']}"
    )

    # And with the default policy, the run fails outright rather than
    # returning a plausible-looking model answer.
    with pytest.raises(Exception):
        asyncio.run(
            orchestrator.execute_yaml(
                yaml_content=_pipeline("gernate", "{prompt: hello}"), context={}
            )
        )


def test_an_explicit_auto_instruction_is_still_allowed(control_system):
    """Removing the implicit fallback must not remove the explicit feature.

    `<AUTO>...</AUTO>` is an author deliberately asking the model to interpret
    an instruction. That is not the same as a typo falling through.
    """
    assert is_known_action("<AUTO>summarise the findings</AUTO>")
    assert match_action_family("<AUTO>summarise the findings</AUTO>").name == "auto"


@pytest.mark.parametrize("prose", ['echo "starting"', "write the following content to report.md"])
def test_declared_prose_families_are_still_allowed(prose):
    """Prose families are declared, not arbitrary. They keep working."""
    assert is_known_action(prose)
    assert match_action_family(prose) is not None


# ---------------------------------------------------------------------------
# 4. required parameters are checked at compile time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec", [s for s in ACTION_SPECS if s.required_parameters], ids=lambda s: s.name
)
def test_missing_required_parameter_is_reported(spec, validator):
    """Exit 2, not a mystery at dispatch."""
    result = validator.validate_pipeline_tools(
        {"steps": [{"id": "s", "action": spec.name, "parameters": {}}]}
    )

    missing = [e for e in result.errors if e.error_type == "missing_parameter"]
    assert {e.parameter_name for e in missing} == set(spec.required_parameters), (
        f"expected every one of {sorted(spec.required_parameters)} to be reported, "
        f"got {[e.parameter_name for e in missing]}"
    )


def test_supplying_required_parameters_validates_cleanly(validator):
    result = validator.validate_pipeline_tools(
        {"steps": [{"id": "s", "action": "generate", "parameters": {"prompt": "hi"}}]}
    )

    assert not [e for e in result.errors if e.error_type == "missing_parameter"]


# ---------------------------------------------------------------------------
# 5. result envelopes match their declared schemas
# ---------------------------------------------------------------------------

def test_generate_result_matches_its_declared_schema():
    """The spec's result_schema is a promise, so it is checked against a run."""
    spec = resolve_action("generate")
    orchestrator = create_test_orchestrator()

    result = asyncio.run(
        orchestrator.execute_yaml(
            yaml_content=_pipeline("generate", "{prompt: summarise this}"), context={}
        )
    )

    jsonschema.validate(instance=result["only_step"], schema=spec.result_schema)


def test_structured_result_matches_its_declared_schema():
    spec = resolve_action("generate_structured")
    orchestrator = create_test_orchestrator()

    result = asyncio.run(
        orchestrator.execute_yaml(
            yaml_content=_pipeline(
                "generate_structured",
                '{prompt: extract, schema: {type: object, properties: {test_output: {type: string}}}}',
            ),
            context={},
        )
    )

    jsonschema.validate(instance=result["only_step"], schema=spec.result_schema)


# ---------------------------------------------------------------------------
# 6. the registry itself is well formed
# ---------------------------------------------------------------------------

def test_no_spelling_is_claimed_twice():
    """Enforced at import; asserted here so the reason is recorded."""
    seen = set()
    for spec in ACTION_SPECS:
        for spelling in (spec.name, *spec.aliases):
            assert spelling.lower() not in seen, f"duplicate spelling {spelling!r}"
            seen.add(spelling.lower())


def test_requires_is_constrained():
    with pytest.raises(ValueError, match="requires must be"):
        ActionSpec(name="bogus", summary="x", requires="magic")


def test_a_spec_cannot_alias_itself():
    with pytest.raises(ValueError, match="its own name as an alias"):
        ActionSpec(name="bogus", summary="x", aliases=frozenset({"bogus"}))


def test_every_spec_has_a_summary_for_the_generated_docs():
    for spec in ACTION_SPECS:
        assert spec.summary.strip(), f"{spec.name} has no summary"
    for family in ACTION_FAMILIES:
        assert family.summary.strip(), f"{family.name} has no summary"


# ---------------------------------------------------------------------------
# 7. the tool forms are unaffected by the action contract
#
# Folded in from tests/test_builtin_actions.py (#241), whose remaining
# assertions duplicated the ones above once the registry became authoritative.
# ---------------------------------------------------------------------------

def test_the_legacy_single_field_form_still_resolves_as_a_tool(validator):
    """`action: filesystem` names the tool itself and keeps full validation.

    `filesystem` is both a registered tool and a registered action name. The
    tool lookup has to win, or this form silently loses its parameter checks.
    """
    result = validator.validate_pipeline_tools(
        {"steps": [{"id": "s", "action": "filesystem", "parameters": {}}]}
    )

    assert result.tool_availability.get("filesystem") is True


def test_the_two_field_form_resolves_tool_never_action(validator):
    result = validator.validate_pipeline_tools(
        {
            "steps": [
                {
                    "id": "s",
                    "tool": "filesystem",
                    "action": "read",
                    "parameters": {"path": "x.txt"},
                }
            ]
        }
    )

    assert "filesystem" in result.tool_availability
    assert "read" not in result.tool_availability, (
        "`read` is an operation on the tool, not a tool to look up"
    )


def test_a_builtin_action_dispatches_away_from_the_model(control_system):
    """`evaluate_condition` is a marker action with no model involvement.

    Reaching the model would produce prose instead of a verdict, which is the
    boundary the whole contract exists to hold.
    """
    task = Task(
        id="check",
        name="check",
        action="evaluate_condition",
        parameters={"condition": "1 == 1"},
    )
    result = asyncio.run(control_system._execute_task_impl(task, {}))

    assert isinstance(result, dict)
    assert "result" in result or "condition" in result

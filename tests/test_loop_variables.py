"""A loop variable is a loop variable in both spellings, in every loop.

`{{ item.name }}` inside a `for_each` step is bound by the runtime. The
data-flow validator accepted it; the template validator knew only the
`$`-prefixed spelling::

    self.loop_vars = {'$item', '$index', '$is_first', '$is_last', ...}

so the bare form -- the one every example in the catalogue actually uses --
was reported as `Undefined variable: 'item'` (#469). That is the false
positive class removed in #448 (`slugify`), #450 (`now`), #454
(`execution.timestamp`) and #459 (`pipeline_id`): a name the runtime provides,
rejected by validation.

Adding the bare names then exposed a second disagreement. The walker decided a
step was a loop with ``'for_each' in obj or 'while' in obj``, so a step
written with ``foreach`` -- an alias the compiler accepts -- was not a loop as
far as the validator was concerned, and its loop variables became "used
outside of loop context". The message changed; the pipeline was still wrongly
rejected.

Both name sets are now declared once, in `core.template_globals`, with the
rest of the pipeline language.
"""

import subprocess
import sys
import os
from pathlib import Path

import pytest

from orchestrator.core.loop_contracts import FOR_EACH, LOOP_CONTRACTS
from orchestrator.core.template_globals import (
    ALL_LOOP_VARIABLES,
    DOLLAR_LOOP_VARIABLES,
    LOOP_STEP_KEYS,
    LOOP_VARIABLES,
)
from orchestrator.validation.template_validator import TemplateValidator

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).resolve().parent.parent


def _check(template, context=None, in_loop=False):
    """`in_loop` means a `for_each` specifically.

    Passing `True` would admit the union of every construct's bindings,
    which is what let `{{ is_last }}` pass inside a `while` loop. Tests
    that go through the imprecise path cannot notice when it is wrong.
    """
    bindings = FOR_EACH.all_bindings() if in_loop else frozenset()
    result = TemplateValidator().validate_template(
        template, context or {}, None, [], bindings
    )
    return result.is_valid, [error.error_type for error in result.errors]


# ---------------------------------------------------------------------------
# The bare spelling is the one people write
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FOR_EACH.bindings))
def test_every_loop_variable_is_accepted_inside_a_loop(name):
    """`FOR_EACH.bindings`, not the union: a `for_each` binds `item`, not
    `queue_size`, and asserting over the union would pass for the wrong
    reason."""
    valid, errors = _check("{{ %s }}" % name, in_loop=True)
    assert valid, f"{name} is bound inside a loop and was rejected: {errors}"


def test_an_attribute_of_a_loop_variable_is_accepted():
    """`{{ item.name }}` is how a loop over objects is actually written."""
    assert _check("{{ item.name }}", in_loop=True)[0]


def test_a_loop_variable_is_still_refused_outside_a_loop():
    """The gate that makes accepting the bare names safe.

    Without it, adding these names would trade a loud false positive for a
    silent false negative -- the trade rejected in #461.
    """
    valid, errors = _check("{{ item.name }}", in_loop=False)
    assert not valid
    assert "loop_variable_outside_loop" in errors, errors


def test_a_declared_input_named_item_wins():
    """A pipeline may name a parameter after a loop word.

    The loop check runs before the context lookup, so without this a pipeline
    declaring `item` would be told its own parameter is a misplaced loop
    variable.
    """
    assert _check("{{ item }}", {"item": "a declared input"}, in_loop=False)[0]


@pytest.mark.parametrize("name", ["item_count", "items", "indexed", "iterations"])
def test_a_name_that_merely_contains_a_loop_word_is_not_one(name):
    """The `$` spellings are matched as raw text, because Jinja cannot parse
    `$item`. Applying that substring scan to the bare names would match `item`
    inside `items` -- the text-matching class of bug #458 removed."""
    assert _check("{{ %s }}" % name, {name: 1}, in_loop=False)[0]


def test_the_dollar_scan_covers_only_the_dollar_spellings():
    """A guard on the above, at the source rather than through behaviour."""
    assert all(name.startswith("$") for name in DOLLAR_LOOP_VARIABLES)
    assert ALL_LOOP_VARIABLES == LOOP_VARIABLES | DOLLAR_LOOP_VARIABLES


# ---------------------------------------------------------------------------
# Every way of writing a loop counts as one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", LOOP_STEP_KEYS)
def test_every_loop_step_key_establishes_loop_context(key):
    """`foreach` is an alias the compiler accepts. A validator that knows only
    `for_each` rejects a pipeline the compiler is happy to run.

    Each construct is probed with a name it actually binds. The earlier
    version of this test used `item` for all five, which only passed because
    every construct saw the union -- `while` and `action_loop` bind no `item`.
    Requiring full validity is what exposed that; the weaker "no error
    mentions `item`" assertion had been satisfied by a pipeline that was
    rejected for an unrelated reason.
    """
    contract = LOOP_CONTRACTS[key]
    bound = sorted(contract.bindings)[0]
    result = TemplateValidator().validate_pipeline_templates(
        {
            "id": "p",
            "steps": [{
                "id": "looped",
                key: "{{ some_source }}",
                "parameters": {"text": "{{ %s }}" % bound},
            }],
        },
        {"some_source": [1]},
    )
    assert result.is_valid, (
        f"a step declaring '{key}' binds {bound}, but the step was rejected: "
        f"{[(e.error_type, e.message) for e in result.errors]}"
    )


def test_jinja_s_own_loop_object_is_not_confused_with_ours():
    """`{% for r in xs %}{{ loop.index }}{% endfor %}` binds `loop` itself.

    Jinja's own analysis already excludes it, so `loop` being in our set must
    not start reporting it.
    """
    assert _check(
        "{% for r in xs %}{{ loop.index }}{% endfor %}", {"xs": []}, in_loop=False
    )[0]


# ---------------------------------------------------------------------------
# One declaration
# ---------------------------------------------------------------------------

def test_both_validators_use_the_same_declaration():
    """Two sets of loop names is what produced #469.

    Identity, not equality: an equal copy can drift, and the point is that
    there is nothing to drift from.
    """
    from orchestrator.validation.data_flow_validator import LOOP_VARIABLES as data_flow

    assert data_flow is LOOP_VARIABLES


def test_the_template_validator_uses_the_shared_set():
    assert TemplateValidator().loop_vars == ALL_LOOP_VARIABLES


# ---------------------------------------------------------------------------
# The catalogue files this was found in
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.parametrize(
    "example", ["examples/fact_checker.yaml", "examples/enhanced/fact_checker_enhanced.yaml"]
)
def test_the_examples_that_exposed_this_now_validate(example):
    """Found by triaging the catalogue's largest failure group rather than by
    reading the code."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ORCHESTRATOR_AUTO_INSTALL"] = "0"
    result = subprocess.run(
        [sys.executable, "-m", "orchestrator.cli", "validate", example],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-600:] + result.stderr[-600:]

"""The loop contracts are checked against a running pipeline, not against
themselves.

`tests/test_loop_contracts.py` asks whether the validator agrees with
`core.loop_contracts`. Both sides of that question come from the same table,
so it proves the table is applied consistently and nothing at all about
whether the table is *true*. That gap is how #473 shipped a contract declaring
`foreach` an alias of `for_each` -- consistent everywhere, and false: no engine
expands it (#475). It is also how `action_loop`'s body came to be validated
outside the loop it is the body of.

So this module executes pipelines. For every name a contract claims, a real
run must render it; for names withheld, a real run must leave them unrendered.
The table cannot drift from the runtime without a failure here.

No model is involved: every pipeline writes files with the filesystem tool, so
these are deterministic and hermetic.
"""

import asyncio
import re
from pathlib import Path

import pytest

from orchestrator.core.loop_contracts import (
    ACTION_LOOP,
    CREATE_PARALLEL_QUEUE,
    FOR_EACH,
    WHILE,
)
from orchestrator.validation.template_validator import TemplateValidator
from tests.test_infrastructure import create_test_orchestrator

pytestmark = [pytest.mark.contract, pytest.mark.e2e]


#: Each construct in the shape that actually runs, with `{probe}` where the
#: name under test is substituted and `{out}` for the run directory. The
#: shapes are not interchangeable and were established by execution:
#: `create_parallel_queue` needs the `action:` key beside its block, and the
#: actions inside it cannot use `tool:`.
BODY_PIPELINES = {
    FOR_EACH.key: """
id: probe_for_each
name: probe
steps:
  - id: loop
    for_each: "['A', 'B']"
    steps:
      - id: emit
        tool: filesystem
        action: write
        parameters:
          path: "{out}/probe.txt"
          content: "<{{{{ {probe} }}}}>"
""",
    WHILE.key: """
id: probe_while
name: probe
steps:
  - id: loop
    while: "{{{{ iteration < 1 }}}}"
    max_iterations: 2
    steps:
      - id: emit
        tool: filesystem
        action: write
        parameters:
          path: "{out}/probe.txt"
          content: "<{{{{ {probe} }}}}>"
""",
    ACTION_LOOP.key: """
id: probe_action_loop
name: probe
steps:
  - id: loop
    action_loop:
      - action: filesystem
        parameters:
          action: write
          path: "{out}/probe.txt"
          content: "<{{{{ {probe} }}}}>"
    until: "{{{{ iteration >= 1 }}}}"
    max_iterations: 1
""",
    CREATE_PARALLEL_QUEUE.key: """
id: probe_queue
name: probe
steps:
  - id: loop
    action: create_parallel_queue
    create_parallel_queue:
      "on": "['A']"
      action_loop:
        - action: filesystem
          parameters:
            action: write
            path: "{out}/probe.txt"
            content: "<{{{{ {probe} }}}}>"
""",
}

CONTRACTS = {c.key: c for c in (FOR_EACH, WHILE, ACTION_LOOP, CREATE_PARALLEL_QUEUE)}

#: Names each construct must NOT bind, chosen from what a *different*
#: construct binds so the check is about this construct rather than about the
#: name being unknown everywhere.
WITHHELD = {
    FOR_EACH.key: ["queue_size", "termination_reason"],
    WHILE.key: ["queue", "has_previous"],
    ACTION_LOOP.key: ["item", "queue_size"],
    CREATE_PARALLEL_QUEUE.key: ["iteration", "total_duration"],
}


def _run(pipeline: str, out: Path) -> str:
    """Run a probe pipeline and return what reached the file.

    An empty string means the step never wrote, which is how an unresolved
    reference surfaces here -- `create_parallel_queue` reports success while
    skipping the write (#478), so the artifact is the evidence, not the
    result payload.
    """
    try:
        asyncio.run(create_test_orchestrator().execute_yaml(pipeline, {}))
    except Exception as exc:  # a construct that cannot run is a real failure
        return f"!{type(exc).__name__}: {exc}"
    probe = out / "probe.txt"
    return probe.read_text() if probe.exists() else ""


def _rendered(written: str, name: str) -> bool:
    """Whether `{{ name }}` became a value rather than passing through."""
    if not written or written.startswith("!"):
        return False
    body = re.fullmatch(r"<(.*)>", written, re.S)
    if body is None:
        return False
    value = body.group(1)
    return value != "" and "{{" not in value


@pytest.mark.parametrize(
    "key,name",
    [(key, name) for key, c in CONTRACTS.items() for name in sorted(c.bindings)],
)
def test_every_declared_binding_renders_in_a_real_run(key, name, tmp_path):
    """A contract that claims a name the runtime does not bind is a false
    acceptance: validation passes and the pipeline fails to render."""
    written = _run(BODY_PIPELINES[key].format(out=tmp_path, probe=name), tmp_path)
    assert _rendered(written, name), (
        f"{key} declares '{name}' but a real run produced {written!r}"
    )


@pytest.mark.parametrize(
    "key,name",
    [(key, name) for key, names in WITHHELD.items() for name in names],
)
def test_a_withheld_name_does_not_render(key, name, tmp_path):
    """The other direction: a name the contract omits must genuinely be
    absent, or omitting it is a false rejection of a working pipeline."""
    written = _run(BODY_PIPELINES[key].format(out=tmp_path, probe=name), tmp_path)
    assert not _rendered(written, name), (
        f"{key} withholds '{name}' but a real run rendered it as {written!r}, "
        f"so validation rejects a pipeline that works"
    )


@pytest.mark.parametrize(
    "key,name",
    [(key, name) for key, c in CONTRACTS.items() for name in sorted(c.bindings)],
)
def test_validation_accepts_every_name_the_runtime_binds(key, name, tmp_path):
    """The two surfaces meet here: what runs must also validate."""
    result = TemplateValidator().validate_pipeline_templates(
        {"id": "p", "steps": [{
            "id": "s", key: _minimal_construct_value(key),
            "parameters": {"t": "{{ %s }}" % name},
        }]},
        {},
    )
    loop_errors = [
        e for e in result.errors
        if e.error_type.startswith("loop_variable")
    ]
    assert not loop_errors, (
        f"{key} binds '{name}' at run time but validation rejected it: "
        f"{[(e.error_type, e.message) for e in loop_errors]}"
    )


def _minimal_construct_value(key):
    """The smallest value that makes a step declare `key`."""
    if key == ACTION_LOOP.key:
        return [{"action": "noop"}]
    if key == CREATE_PARALLEL_QUEUE.key:
        return {"on": "['A']", "action_loop": [{"action": "noop"}]}
    return "['A']"


def test_the_while_condition_can_read_its_own_iteration_count():
    """The condition is re-evaluated per iteration, so `iteration` is bound
    there -- unlike a `for_each` iterable, which resolves once, before there
    is anything to bind. #473 modelled both as the same kind of field and
    rejected this pipeline, which runs."""
    result = TemplateValidator().validate_pipeline_templates(
        {"id": "p", "steps": [{
            "id": "s",
            "while": "{{ iteration < 3 }}",
            "steps": [{"id": "b", "parameters": {"t": "x"}}],
        }]},
        {},
    )
    assert result.is_valid, [(e.error_type, e.message) for e in result.errors]


def test_the_while_condition_stops_the_loop_at_the_iteration_it_names(tmp_path):
    """And it is bound in the sense that matters: the count controls the run."""
    pipeline = BODY_PIPELINES[WHILE.key].replace(
        'while: "{{{{ iteration < 1 }}}}"', 'while: "{{{{ iteration < 3 }}}}"'
    ).replace("max_iterations: 2", "max_iterations: 9").replace(
        '"{out}/probe.txt"', '"{out}/{{{{ iteration }}}}.txt"'
    )
    _run(pipeline.format(out=tmp_path, probe="iteration"), tmp_path)
    written = sorted(p.name for p in tmp_path.glob("*.txt"))
    assert written == ["0.txt", "1.txt", "2.txt"], (
        f"the condition names iteration < 3, so the body runs three times; "
        f"got {written}"
    )

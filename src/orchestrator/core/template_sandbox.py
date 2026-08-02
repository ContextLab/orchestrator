"""One place that builds Jinja environments, and it builds sandboxed ones.

A pipeline's `{{ }}` expressions are authored with the pipeline. The values
substituted into them are not: they arrive from `-i name=value`, from an
inputs file, or from an upstream step's output. Those values are themselves
rendered, which is a deliberate feature -- `-i out_dir='{{ base }}/reports'`
is useful -- but it means a parameter value is executed as a template.

Jinja's stock `Environment` executes it with Python's object graph in reach.
`orchestrator run 01_hello_filesystem.yaml -i greeting='{{ "".__class__ }}'`
was enough to walk from a string literal to `__subclasses__()`, which is the
first hop of the standard Jinja sandbox escape. `SandboxedEnvironment` refuses
that traversal while leaving ordinary expressions -- arithmetic, filters,
`{{ step.result.content }}` -- working exactly as before.

Every environment on the execution path is built here so a plain
`Environment(...)` cannot quietly reappear in one of them; `test_template_sandbox.py`
asserts that none does.
"""

from __future__ import annotations

from typing import Any

from jinja2 import StrictUndefined, Template
from jinja2.sandbox import SandboxedEnvironment


def create_sandboxed_environment(**kwargs: Any) -> SandboxedEnvironment:
    """A Jinja environment that will not hand out Python internals.

    `undefined=StrictUndefined` unless the caller overrides it: an unresolved
    name must fail the render rather than silently become an empty string.
    """
    kwargs.setdefault("undefined", StrictUndefined)
    return SandboxedEnvironment(**kwargs)


def sandboxed_template(source: str, **kwargs: Any) -> Template:
    """A single template, compiled under the sandbox.

    For the call sites that want one template rather than a whole environment.
    `Template(source)` there would build its own unsandboxed environment.
    """
    return create_sandboxed_environment(**kwargs).from_string(source)

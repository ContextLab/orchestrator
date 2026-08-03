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

from functools import cache
from typing import Any, FrozenSet

from jinja2 import StrictUndefined, Template
from jinja2.sandbox import SandboxedEnvironment


def create_sandboxed_environment(**kwargs: Any) -> SandboxedEnvironment:
    """A Jinja environment that will not hand out Python internals.

    `undefined=StrictUndefined` unless the caller overrides it: an unresolved
    name must fail the render rather than silently become an empty string.
    """
    kwargs.setdefault("undefined", StrictUndefined)
    return SandboxedEnvironment(**kwargs)


def create_pipeline_environment(**kwargs: Any) -> SandboxedEnvironment:
    """A sandboxed environment that also knows the orchestrator's own filters.

    `create_sandboxed_environment` gives a bare Jinja environment. Anything
    that renders *pipeline* templates needs more than that: `slugify`,
    `basename`, `from_json` and the rest are part of the pipeline language, and
    an environment without them reports a working pipeline as broken.

    Three environments used to build their own filter sets independently -- the
    template validator (56 filters), the YAML compiler (61) and the runtime
    (70). A pipeline using `{{ title | slugify }}` therefore failed validation
    and compiled fine, or compiled and failed on `truncate_words`, depending on
    which one it met first. They are all built here now, from the runtime's
    registry, so the three cannot drift apart again.

    Filters are copied; **globals deliberately are not**. See
    `pipeline_global_names` for why.
    """
    from .template_manager import TemplateManager

    env = create_sandboxed_environment(**kwargs)
    # TemplateManager owns the filter set; this copies it rather than
    # re-declaring it, which is what let the three sets diverge.
    env.filters.update(TemplateManager().env.filters)
    return env


@cache
def pipeline_global_names() -> FrozenSet[str]:
    """The global functions the pipeline language offers: `now()` and friends.

    Filters transform a value the caller already has, so any environment can
    evaluate one. Globals do not: every one of ours answers a question about
    *the state of a run* -- what time it is, whether a file exists yet, which
    loop iteration this is. Evaluating them anywhere but the runtime gives a
    confidently wrong answer rather than an error::

        - id: make_it     # writes ./artifact
        - id: check_it    # content: "{{ file_exists('artifact') }}"

    `check_it` renders `True`, because by then `make_it` has run. Copy the
    globals into the compiler's environment and it renders at compile time
    instead, when the file does not exist, and writes `False`. So the compiler
    and the validators must know these *names* without being able to call
    them: an unregistered global raises, and the compiler keeps the unrendered
    template for the runtime, which is exactly the behaviour we want.

    That left the validators reporting `Undefined variable: 'now'` on
    pipelines that run correctly -- the false-positive class of #448 again.
    They import this instead, so the names are stated once.

    The names come from `template_globals.GLOBAL_SPECS`, which also carries
    each one's argument contract. They were originally derived from what
    `TemplateManager` happened to register, which could not drift but could not
    express an arity either; `test_template_globals.py` compares the two, so
    the drift protection survives the change.
    """
    from .template_globals import GLOBAL_NAMES

    return GLOBAL_NAMES


def sandboxed_template(source: str, **kwargs: Any) -> Template:
    """A single template, compiled under the sandbox.

    For the call sites that want one template rather than a whole environment.
    `Template(source)` there would build its own unsandboxed environment.
    """
    return create_sandboxed_environment(**kwargs).from_string(source)

"""Control-flow routing between steps.

A step may name where execution goes next:

    - id: check_topic
      action: evaluate_condition
      parameters:
        condition: "{{ topic | length > 10 }}"
      on_false: short_topic_handler
      on_success: main_processing
      on_failure: error_recovery

`on_false` applies when the step's own `condition:` evaluates false, so the
step is skipped. `on_success` and `on_failure` apply to how the step itself
ended. Routing jumps forward: steps between the source and the target are
marked skipped, which is the same machinery `goto` already used.

## The `on_failure` collision

`on_failure` already existed, meaning a *failure policy*: one of `fail`,
`continue`, `skip`, `retry`. #333 asks for the same key to name a step to jump
to. Both readings are useful and both are now supported, disambiguated by
value: a reserved policy word keeps its policy meaning, anything else is a
step id.

That is only safe because the ambiguity is refused rather than guessed at --
naming a step `retry` and routing to it would silently become a policy, so a
pipeline that does so fails at compile time instead.
"""

from __future__ import annotations

from typing import FrozenSet

#: Values of `on_failure` that select a policy rather than a routing target.
FAILURE_POLICIES: FrozenSet[str] = frozenset({"fail", "continue", "skip", "retry"})

#: Step keys that name another step to jump to.
ROUTING_KEYS: FrozenSet[str] = frozenset({"on_false", "on_success", "on_failure"})


def is_failure_policy(value: object) -> bool:
    """Whether an `on_failure` value selects a policy instead of a step."""
    return isinstance(value, str) and value.strip().lower() in FAILURE_POLICIES


def routing_targets(step: dict) -> dict:
    """The routing keys of `step` that name a step id, as {key: target}.

    `on_failure` is included only when its value is not a reserved policy.
    """
    targets = {}
    for key in ("on_false", "on_success"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            targets[key] = value.strip()

    on_failure = step.get("on_failure")
    if isinstance(on_failure, str) and on_failure.strip() and not is_failure_policy(on_failure):
        targets["on_failure"] = on_failure.strip()

    return targets

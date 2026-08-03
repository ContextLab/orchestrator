"""What a run knows about itself: `{{ execution.timestamp }}` and friends.

A pipeline can ask for facts about the run it is part of -- when it started,
which run it is. Seven places built that answer independently, in four
different formats::

    orchestrator.py:307              %Y-%m-%d-%H:%M:%S
    orchestrator.py:1424, :1994      .isoformat()
    control_system.py:222            %Y-%m-%d %H:%M:%S
    hybrid_control_system.py:594     %Y-%m-%d %H:%M:%S
    declarative_engine.py:121        no timestamp at all -- `start_time`

They did not merely disagree between engines. `_execute_level` rebuilt the
dict at *every level of the graph*, overwriting the one the run had already
registered, so a single run answered its own question differently each time::

    step one   ->   2026-08-02T20:01:55.182681
    step two   ->   2026-08-02T20:01:55.184368

Two rows of a report, stamped two thousandths of a second apart, from one
run. Anything using the value to name an output file wrote several.

So the value is computed once, when the run starts, and every later reader
gets that same value. `execution_namespace_for` is how they get it: the first
call on a run's context computes it and stores it there, every later call
returns what it finds. One run, one answer.

The exposed fields are a closed set. `{{ execution.strated_at }}` is a typo,
not a field, and is refused at compile time rather than rendering as an empty
string into somebody's report -- which is what an open namespace would do.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, MutableMapping, Optional, Tuple

#: Where the run's context dict carries its `RuntimeContext`.
CONTEXT_KEY = "_runtime_context"

#: The name a template reaches it by: `{{ execution.timestamp }}`.
RUNTIME_NAMESPACE = "execution"

#: Every field `{{ execution.* }}` may name. `test_runtime_context.py` asserts
#: this matches what `as_template_namespace` actually produces.
EXECUTION_FIELD_NAMES: Tuple[str, ...] = (
    "id",
    "started_at",
    "timestamp",
    "date",
    "time",
)

EXECUTION_FIELDS: FrozenSet[str] = frozenset(EXECUTION_FIELD_NAMES)


@dataclass(frozen=True)
class RuntimeContext:
    """One run's identity and start time. Immutable, created once."""

    id: str
    started_at: datetime

    @classmethod
    def create(cls, execution_id: Optional[str] = None) -> "RuntimeContext":
        """A context for a run starting now, in UTC.

        UTC rather than local time so two machines running the same pipeline
        produce comparable stamps, and so a run spanning a daylight-saving
        change does not go backwards.
        """
        return cls(
            id=execution_id or f"run-{uuid.uuid4().hex[:12]}",
            started_at=datetime.now(timezone.utc),
        )

    @property
    def timestamp(self) -> str:
        """`started_at` under its older name.

        59 of the catalogue's 61 `execution.*` references spell it this way.
        It is the same instant, not a second reading of the clock.
        """
        return self.started_at.isoformat()

    def as_template_namespace(self) -> Dict[str, str]:
        """What `{{ execution }}` resolves to.

        Plain strings in a plain dict, because a run's context is checkpointed
        as JSON: caching a `RuntimeContext` itself made every checkpointed run
        fail with "Object of type RuntimeContext is not JSON serializable".
        """
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "timestamp": self.timestamp,
            "date": self.started_at.strftime("%Y-%m-%d"),
            "time": self.started_at.strftime("%H:%M:%S"),
        }


def execution_namespace_for(context: MutableMapping[str, Any]) -> Dict[str, str]:
    """The run's answer, computed on the first ask and reused after.

    Storing it back on the run's context is what makes "one run, one answer"
    hold without threading an instance through every caller -- and every
    caller already holds the run's context dict.
    """
    cached = context.get(CONTEXT_KEY)
    if isinstance(cached, dict) and EXECUTION_FIELDS <= set(cached):
        return cached

    namespace = RuntimeContext.create(context.get("execution_id")).as_template_namespace()
    context[CONTEXT_KEY] = namespace
    return namespace

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

import time
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

#: Names the runtime registers directly, without the `execution.` prefix.
#: They render correctly today and were reported as undefined variables, which
#: is the same false positive `execution.timestamp` had before it was declared.
#: `timestamp` is the run's start time, so it must be *the same instant* as
#: `execution.timestamp` -- it used to be a separate reading of the clock, in a
#: different format, which is the divergence this module exists to end.
BARE_RUNTIME_NAMES: FrozenSet[str] = frozenset(
    {"pipeline_id", "execution_id", "timestamp"}
)


def new_execution_id(pipeline_id: Optional[str] = None) -> str:
    """An identifier no two runs share.

    This was ``f"{pipeline.id}_{int(time.time())}"``, which is unique only
    until the same pipeline is started twice inside one second -- routine
    under a test suite, a scheduler, or a retry loop. Two runs then shared an
    id, and since that id names checkpoints and stamps artifacts, the second
    run resumed from the first one's checkpoint.

    The second-resolution stamp is kept because it sorts and reads well; the
    uniqueness comes from the entropy after it.
    """
    stamp = int(time.time())
    entropy = uuid.uuid4().hex[:8]
    if pipeline_id:
        return f"{pipeline_id}_{stamp}_{entropy}"
    return f"run-{stamp}-{entropy}"


@dataclass(frozen=True)
class RuntimeContext:
    """One run's identity and start time. Immutable, created once.

    Created by whoever owns the run -- an orchestrator, an engine, a
    `PipelineExecutionState`. Context dictionaries receive a *projection* of
    it (see `project_into`) and never establish it themselves, because a
    dictionary cannot know whether it is the run or merely a copy of part of
    it. `PipelineExecutionState.get_available_context` proved the difference:
    it built a fresh dict per call and let the identity be established inside
    it, so every read of the run's context invented a new run.
    """

    id: str
    started_at: datetime
    pipeline_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        execution_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
    ) -> "RuntimeContext":
        """A context for a run starting now, in UTC.

        UTC rather than local time so two machines running the same pipeline
        produce comparable stamps, and so a run spanning a daylight-saving
        change does not go backwards.
        """
        return cls(
            id=execution_id or new_execution_id(pipeline_id),
            started_at=datetime.now(timezone.utc),
            pipeline_id=pipeline_id,
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


    def public_names(self) -> Dict[str, Any]:
        """Exactly the names the pipeline language declares, and no others.

        One builder for every engine. Each used to assemble its own idea of
        what a run offers, so `{{ execution.timestamp }}` worked under the
        main orchestrator and not the declarative engine, `{{ timestamp }}`
        the reverse, and two engines offered a `pipeline` namespace that
        validation refuses. The language a pipeline is written against must
        not depend on which engine happens to run it.
        """
        namespace = self.as_template_namespace()
        names: Dict[str, Any] = {
            RUNTIME_NAMESPACE: namespace,
            "execution_id": self.id,
            "timestamp": namespace["started_at"],
        }
        if self.pipeline_id is not None:
            names["pipeline_id"] = self.pipeline_id
        return names

    def project_into(self, context: MutableMapping[str, Any]) -> Dict[str, str]:
        """Write this run's public names into a context dict.

        Idempotent: the values come from an immutable object, so projecting
        twice writes the same thing twice. Returns the `execution` namespace
        for callers that want it directly.
        """
        namespace = self.as_template_namespace()
        context[CONTEXT_KEY] = namespace
        context.update(self.public_names())
        return namespace


def runtime_context_for(context: MutableMapping[str, Any]) -> RuntimeContext:
    """The run a context dict belongs to, established on first ask.

    For callers that hold a run's context but not the object -- a control
    system, an engine mid-run. The identity is stored back on the dict, so
    every later caller holding that same dict gets the same run.

    `PipelineExecutionState` deliberately does not use this: it *is* the run
    owner and holds a `RuntimeContext` directly.
    """
    cached = context.get(CONTEXT_KEY)
    if isinstance(cached, dict) and EXECUTION_FIELDS <= set(cached):
        return RuntimeContext(
            id=cached["id"],
            started_at=datetime.fromisoformat(cached["started_at"]),
            pipeline_id=context.get("pipeline_id"),
        )

    return RuntimeContext.create(
        execution_id=context.get("execution_id"),
        pipeline_id=context.get("pipeline_id"),
    )


def execution_namespace_for(context: MutableMapping[str, Any]) -> Dict[str, str]:
    """The run's answer, computed on the first ask and reused after.

    Storing it back on the run's context is what makes "one run, one answer"
    hold without threading an instance through every caller -- and every
    caller already holds the run's context dict.
    """
    return runtime_context_for(context).project_into(context)

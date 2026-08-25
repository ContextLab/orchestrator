"""Event vocabulary for the sherpa kernel (issue #492, single append-only log).

The event log is the source of truth; every other structure is a projection.
Kinds are a closed vocabulary validated by :class:`Store.append` — see
``sherpa.store``.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

EVENT_KINDS = frozenset(
    {
        "run_started",
        "run_terminal",
        "node_created",
        "node_state_changed",
        "lease_acquired",
        "lease_released",
        # A worker was refused a node another live session holds. Recorded so
        # that exactly-once execution is auditable, not merely asserted.
        "lease_denied",
        "attempt_started",
        "attempt_finished",
        "admission_checked",
        "tool_call_started",
        "tool_call_finished",
        "artifact_written",
        "message_enqueued",
        "message_delivered",
        "journal_appended",
        "chunk_indexed",
        "summary_created",
        "finding_raised",
        "finding_disposition",
        "plan_recorded",
        "node_progress",
        "decompose_outcome",
        "usage_checkpoint",
        "crash_detected",
        "orphan_recovered",
        "review_round",
        "cache_hit",
    }
)


class Event(BaseModel):
    """One immutable entry in the run's causal history."""

    seq: int | None = None
    ts: float = Field(default_factory=time.time)
    run_id: str
    node_key: str | None = None
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    causal_seq: int | None = None

    @field_validator("kind")
    @classmethod
    def _kind_known(cls, v: str) -> str:
        if v not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {v!r}")
        return v

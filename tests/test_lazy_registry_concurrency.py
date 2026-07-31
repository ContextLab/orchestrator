"""Concurrency behaviour of the lazy model registry.

Population is deferred to the first request that genuinely needs a model. The
deferral is what keeps tool-only pipelines from touching credentials, but it
means the first *concurrent* burst of requests all arrive at an empty
registry at once -- so the discovery step has to be serialised.

No mocks: these use a real ``LazyModelRegistry`` with a real populate
callable and real threads.
"""

import threading
import time

import pytest

from orchestrator.core.model import (
    Model,
    ModelCapabilities,
    ModelCost,
    ModelRequirements,
)
from orchestrator.models.lazy_registry import LazyModelRegistry

pytestmark = pytest.mark.unit


class _StubModel(Model):
    """A minimal registrable model. It is never called -- only registered."""

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            provider="test",
            capabilities=ModelCapabilities(supported_tasks=["generate"]),
            requirements=ModelRequirements(),
            cost=ModelCost(is_free=True),
        )

    async def generate(self, prompt, **kwargs):
        raise NotImplementedError

    async def generate_structured(self, prompt, schema, **kwargs):
        raise NotImplementedError

    async def health_check(self) -> bool:
        return True

    async def estimate_cost(self, prompt, max_tokens=None) -> float:
        return 0.0


def test_population_runs_exactly_once_under_concurrent_demand():
    """Twenty threads demanding a model must trigger one discovery, not twenty."""
    calls = []

    def populate(registry):
        calls.append(1)
        time.sleep(0.05)  # discovery is slow: imports, credential reads, probes
        registry.register_model(_StubModel("test:alpha"))

    registry = LazyModelRegistry(populate)
    barrier = threading.Barrier(20)
    errors = []

    def demand():
        barrier.wait()  # maximise the overlap
        try:
            registry.ensure_populated()
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    threads = [threading.Thread(target=demand) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent demand raised: {errors}"
    assert sum(calls) == 1, "discovery must run exactly once"


def test_a_concurrent_caller_never_sees_a_half_populated_registry():
    """The regression this lock exists for.

    The `_populated` flag is set *before* populate() runs, so that a failing
    discovery is not retried on every request. Without a lock, a second thread
    read that flag the instant it flipped and proceeded against a registry
    that was still filling -- reporting "model not found" for a model that was
    about to exist.
    """
    def populate(registry):
        time.sleep(0.05)
        registry.register_model(_StubModel("test:alpha"))

    registry = LazyModelRegistry(populate)
    observations = []
    start = threading.Event()

    def observer():
        start.wait()
        time.sleep(0.01)  # land inside the populate() window
        registry.ensure_populated()
        # By the time ensure_populated() returns, discovery is complete.
        observations.append(len(registry.models))

    threads = [threading.Thread(target=observer) for _ in range(8)]
    for t in threads:
        t.start()
    start.set()
    registry.ensure_populated()
    for t in threads:
        t.join()

    assert observations, "observer threads recorded nothing"
    assert all(count == 1 for count in observations), (
        f"a thread saw a partially populated registry: {observations}"
    )


def test_a_failing_discovery_is_not_retried_on_every_request():
    """The property the flag-before-populate ordering exists to preserve.

    Retrying a failing discovery on each request would re-read credentials and
    re-probe the machine on every single model lookup.
    """
    attempts = []

    def populate(registry):
        attempts.append(1)
        raise RuntimeError("no providers configured")

    registry = LazyModelRegistry(populate)

    with pytest.raises(RuntimeError):
        registry.ensure_populated()
    # Subsequent calls must not retry -- the failure already surfaced.
    registry.ensure_populated()
    registry.ensure_populated()

    assert sum(attempts) == 1, "a failed discovery must not be retried per call"


def test_reentrant_population_does_not_deadlock():
    """Registration paths can reach back into a demand point on the same thread.

    A plain Lock would deadlock there; the registry uses an RLock.
    """
    def populate(registry):
        # Re-entry from within populate(), on the same thread.
        registry.ensure_populated()
        registry.register_model(_StubModel("test:alpha"))

    registry = LazyModelRegistry(populate)

    done = threading.Event()

    def run():
        registry.ensure_populated()
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert done.wait(timeout=5), "re-entrant population deadlocked"
    assert len(registry.models) == 1

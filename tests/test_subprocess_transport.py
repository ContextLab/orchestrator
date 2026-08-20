"""A tool that starts a child process closes the pipes it gave it.

`kill()` and `wait()` reap the *process*, not its pipes. `_reap` did both to a
child that outran its timeout, and the child was duly gone -- `returncode=-9`
-- with its stdout and stderr transports still open.

CPython then reports them from `__del__`, after the loop that owned them is
gone:

    ResourceWarning: unclosed transport <_UnixReadPipeTransport fd=18 open>

The collector decides when that happens, so the failure lands on whatever test
is running at the time. It surfaced as seven failures in `test_failure_policy`
and `test_supported_examples` -- both of which turn a ResourceWarning into an
error deliberately, to catch exactly this -- on Linux/py3.13 only, and named
no code that had anything to do with it. Tracemalloc put the allocation at
`system_tools.py`'s `create_subprocess_shell`.

So these assert the invariant at its source rather than waiting for a
collector somewhere else to notice: after the tool returns, the transport it
opened is closed. That holds on every platform, including the ones where
CPython happened to tidy up on its own and hid the bug.

The subprocesses here are real; only the reference to the `Process` object is
captured, because the tool does not return it.
"""

import asyncio

import pytest

import orchestrator.tools.system_tools as system_tools
from orchestrator.tools.system_tools import TerminalTool

pytestmark = [pytest.mark.contract]


@pytest.fixture
def started_processes(monkeypatch):
    """Every child the tool starts, so its transport can be inspected after.

    A pass-through recorder, not a substitute: the real
    `create_subprocess_shell` runs and a real process is spawned.
    """
    seen = []
    real = asyncio.create_subprocess_shell

    async def recording(*args, **kwargs):
        process = await real(*args, **kwargs)
        seen.append(process)
        return process

    monkeypatch.setattr(system_tools.asyncio, "create_subprocess_shell", recording)
    return seen


def _open_pipes(process):
    """Pipe transports still open on this child's subprocess transport."""
    transport = getattr(process, "_transport", None)
    if transport is None:
        return []
    pipes = getattr(transport, "_pipes", None) or {}
    return [
        fd for fd, proto in pipes.items()
        if getattr(proto, "pipe", None) is not None and not proto.pipe.is_closing()
    ]


def _assert_closed(seen):
    assert seen, "no child process was started, so nothing was verified"
    for process in seen:
        transport = getattr(process, "_transport", None)
        assert transport is not None, "the process has no transport to close"
        assert transport.is_closing() or getattr(transport, "_closed", False), (
            "the subprocess transport is still open after the tool returned"
        )
        assert _open_pipes(process) == [], (
            f"pipe transports left open: {_open_pipes(process)}"
        )


def test_a_command_that_succeeds_closes_its_pipes(started_processes):
    """Not a case that leaked -- a child that exits on its own has its
    transport closed already. Here to hold that boundary, so a future change
    that starts leaking the ordinary path is caught too."""
    result = asyncio.run(TerminalTool().execute(command="echo hello", timeout=10))
    assert result["success"], result
    _assert_closed(started_processes)


def test_a_command_that_fails_closes_its_pipes(started_processes):
    """A non-zero exit is still a child that exited on its own, so likewise
    not a leaking case."""
    result = asyncio.run(TerminalTool().execute(command="exit 3", timeout=10))
    assert not result["success"], result
    _assert_closed(started_processes)


def test_a_command_that_times_out_closes_its_pipes(started_processes):
    """One of the two that actually leaked: the child is killed, and the pipes
    it was given outlive it."""
    result = asyncio.run(TerminalTool().execute(command="sleep 5", timeout=1))
    assert not result["success"], result
    _assert_closed(started_processes)


def test_a_cancelled_command_closes_its_pipes(started_processes):
    """The other. A *step* timeout cancels the tool rather than the command,
    so `_reap`'s own `await` is cancelled and the close has to survive that."""

    async def cancel_mid_command():
        task = asyncio.create_task(
            TerminalTool().execute(command="sleep 5", timeout=30)
        )
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_mid_command())
    _assert_closed(started_processes)


def test_no_resource_warning_survives_the_run(started_processes, recwarn):
    """The symptom itself, at the place that causes it."""
    import gc

    asyncio.run(TerminalTool().execute(command="echo hello", timeout=10))
    gc.collect()
    leaked = [w for w in recwarn if issubclass(w.category, ResourceWarning)]
    assert not leaked, [str(w.message) for w in leaked]

"""A loop this code creates is a loop this code closes.

Reported twice as "unclosed sockets and an unclosed asyncio event loop during
the supported-example suite" -- intermittent, present when suites ran together
and absent when each ran alone. It resisted reproduction for two sessions
because I looked for it in the test harness. It was in the product.

`TemplateManager.include_file_sync` created an event loop, set it as the
thread's current loop, ran one coroutine on it and never closed it. Both
reported symptoms come from that one defect: a selector loop holds a
socketpair for its self-pipe until it is closed, so the unclosed loop *is* the
unclosed sockets.

Only the first call leaked. After it, `get_event_loop()` succeeded and reused
the leaked loop -- so whether the leaking branch ran at all depended on
whether something earlier had left a loop on the thread. That is exactly the
"passes alone, fails combined" signature.

Three other places created a loop, closed it in a `finally`, and left the
*closed* loop set as current. The next caller to ask `get_event_loop()` got a
closed loop, and `include_file()` renders the resulting error into the
document as a comment rather than raising:

    <!-- File inclusion error: README.md - Event loop is closed -->

A document silently missing an included file, depending on what ran earlier in
the same thread.
"""

import asyncio
import gc
from pathlib import Path

import pytest

from orchestrator.core.file_inclusion import FileInclusionProcessor
from orchestrator.core.template_manager import TemplateManager

pytestmark = [pytest.mark.contract]

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def no_current_loop():
    """A thread with no event loop set, as at the start of a fresh process."""
    try:
        previous = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous = None
    asyncio.set_event_loop(None)
    yield
    asyncio.set_event_loop(previous)


def _open_loops():
    return {
        id(obj)
        for obj in gc.get_objects()
        if isinstance(obj, asyncio.AbstractEventLoop) and not obj.is_closed()
    }


def _render_include(tmp_path, name="included.txt", body="hello from the file"):
    """Render `{{ include_file(...) }}` against a real file.

    The processor only resolves paths inside its `base_dirs`, so it is built
    rooted at `tmp_path` rather than the test writing into the repository.
    """
    (tmp_path / name).write_text(body)
    manager = TemplateManager(
        file_inclusion_processor=FileInclusionProcessor(base_dirs=[str(tmp_path)])
    )
    return manager.env.from_string("{{ include_file('%s') }}" % name).render()


# ---------------------------------------------------------------------------
# Nothing is left open
# ---------------------------------------------------------------------------

def test_including_a_file_leaves_no_open_event_loop(tmp_path, no_current_loop):
    """The leak itself."""
    before = _open_loops()
    _render_include(tmp_path)
    gc.collect()
    leaked = _open_loops() - before

    assert not leaked, (
        f"{len(leaked)} event loop(s) left open by one template render; a "
        f"long-running process renders many"
    )


def test_including_a_file_leaves_no_open_sockets(tmp_path, no_current_loop):
    """The other half of the same report.

    A selector loop holds a socketpair for its self-pipe until closed, so an
    unclosed loop is also two unclosed sockets. Asserting on the sockets
    directly keeps the connection to what was reported.
    """
    before = _open_loops()
    _render_include(tmp_path)
    gc.collect()

    held = [
        sock
        for obj in gc.get_objects()
        if isinstance(obj, asyncio.AbstractEventLoop)
        and id(obj) not in before
        and not obj.is_closed()
        for sock in (getattr(obj, "_csock", None), getattr(obj, "_ssock", None))
        if sock is not None
    ]
    assert not held, f"{len(held)} socket(s) held open by an unclosed loop"


def test_the_thread_is_left_as_it_was_found(tmp_path, no_current_loop):
    """Setting a loop and walking away is what poisoned the next caller."""
    _render_include(tmp_path)

    try:
        current = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        return  # no loop set at all: exactly right

    assert not current.is_closed(), (
        "a closed event loop was left set as the thread's current loop; the "
        "next caller to ask for it gets a loop it cannot run anything on"
    )


# ---------------------------------------------------------------------------
# The cross-contamination that made it a product bug
# ---------------------------------------------------------------------------

def test_a_file_is_included_after_something_else_used_a_loop(tmp_path, no_current_loop):
    """The failure a user would actually see.

    Three call sites closed their loop and left it set. `include_file` then
    found a closed loop, failed, caught its own error and rendered it into the
    document -- so the document was silently missing the file, depending on
    what had run earlier in the same thread.
    """
    stale = asyncio.new_event_loop()
    asyncio.set_event_loop(stale)
    stale.run_until_complete(asyncio.sleep(0))
    stale.close()
    assert asyncio.get_event_loop_policy().get_event_loop().is_closed()

    rendered = _render_include(tmp_path, body="the file content")

    assert "the file content" in rendered, (
        f"the file was not included after a closed loop was left set: "
        f"{rendered!r}"
    )
    assert "Event loop is closed" not in rendered


@pytest.mark.parametrize("repeats", [3])
def test_repeated_includes_do_not_accumulate_loops(tmp_path, no_current_loop, repeats):
    """One leak per call would be worse; one leak ever is still a leak."""
    before = _open_loops()
    for i in range(repeats):
        _render_include(tmp_path, name=f"f{i}.txt", body=f"body {i}")
    gc.collect()

    assert not _open_loops() - before


# ---------------------------------------------------------------------------
# The behaviour that had to be preserved
# ---------------------------------------------------------------------------

def test_including_a_file_still_returns_its_content(tmp_path, no_current_loop):
    assert "hello from the file" in _render_include(tmp_path)


@pytest.mark.asyncio
async def test_inside_a_running_loop_it_declines_instead_of_deadlocking(tmp_path):
    """`asyncio.run` raises if a loop is already running, so the async case
    must still be detected -- and `get_running_loop` is the question actually
    being asked."""
    (tmp_path / "x.txt").write_text("unused")
    rendered = TemplateManager().env.from_string(
        "{{ include_file('x.txt', '%s') }}" % tmp_path
    ).render()

    assert "requires async context" in rendered, (
        f"expected a refusal inside a running loop, got {rendered!r}"
    )


def test_no_call_site_creates_a_loop_it_does_not_hand_to_asyncio_run():
    """A regression guard with a wider net than the four sites fixed.

    `new_event_loop()` paired with `set_event_loop()` is the shape that leaks:
    whoever sets it owns closing it *and* unsetting it, and none of the four
    call sites did both. `asyncio.run` does both, so it is what these should
    use.
    """
    offenders = []
    for path in (REPO / "src" / "orchestrator").rglob("*.py"):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if "new_event_loop()" not in line:
                continue
            window = "\n".join(lines[i : i + 3])
            if "set_event_loop" in window:
                offenders.append(f"{path.relative_to(REPO)}:{i + 1}: {line.strip()}")

    assert not offenders, (
        "these create an event loop and set it as the thread's current loop. "
        "Use asyncio.run, which closes it and unsets it: " + str(offenders)
    )

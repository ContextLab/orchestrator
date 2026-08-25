"""Hermetic tests for channels and capabilities (real files/subprocesses only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sherpa.channel import (
    ChannelRequired,
    EchoChannel,
    LiveChannel,
    RecordedChannel,
    RecordingExhausted,
    RecordingMismatch,
    make_channel,
)
from sherpa.capabilities import (
    AuthorityDenied,
    CapabilityContext,
    CapabilityRegistry,
    FsListDir,
    FsReadFile,
    FsWriteFile,
    PatchError,
    RepoApplyPatch,
    RepoRunTests,
    TextSearchCorpus,
    TextSummarize,
    _parse_unified_diff,
    register_builtins,
    resolve_inputs,
    run_capability,
)
from sherpa.ir import Authority
from sherpa.store import content_hash

pytestmark = [pytest.mark.unit]


def _ctx(store, workspace: Path, granted: Authority | None = None) -> CapabilityContext:
    return CapabilityContext(
        workspace=workspace,
        store=store,
        run_id="r_test",
        node_key="n1",
        channel_factory=lambda: make_channel("recorded", {"summarizer": ["SUMMARY TEXT"]}),
        granted=granted or Authority(fs_read=("**",), fs_write=("**",), subprocess_allow=("**",)),
    )


class TestChannels:
    def test_recorded_fifo_and_exhaustion(self) -> None:
        ch = RecordedChannel({"llm": ["first", "second"]})
        assert ch.complete([], session="llm").text == "first"
        assert ch.complete([], session="llm").text == "second"
        with pytest.raises(RecordingExhausted):
            ch.complete([], session="llm")

    def test_echo_channel_refuses(self) -> None:
        ch = EchoChannel()
        with pytest.raises(ChannelRequired):
            ch.complete([{"role": "user", "content": "hi"}], session="any")

    def test_make_channel_policies(self) -> None:
        assert isinstance(make_channel("recorded"), EchoChannel)
        assert isinstance(make_channel("recorded", {"x": ["y"]}), RecordedChannel)
        assert isinstance(make_channel("live"), LiveChannel)
        with pytest.raises(ValueError):
            make_channel("psychic")

    def test_live_unavailable_raises_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        def boom(name: str, *a: object, **k: object) -> None:
            raise ImportError(f"no {name}")

        monkeypatch.setattr(importlib, "import_module", boom)
        ch = LiveChannel()
        with pytest.raises(Exception):  # ProviderUnavailable or ImportError boundary
            ch.complete([{"role": "user", "content": "x"}], session="s")


class TestRegistry:
    def test_register_and_duplicates(self) -> None:
        reg = CapabilityRegistry()
        register_builtins(reg)
        assert "fs.read_file" in reg.names()
        with pytest.raises(ValueError):
            register_builtins(reg)

    def test_unknown_get(self) -> None:
        reg = CapabilityRegistry()
        with pytest.raises(KeyError):
            reg.get("nope")


class TestFsCapabilities:
    def test_read_write_list_roundtrip(self, store, workspace: Path) -> None:
        ctx = _ctx(store, workspace)
        out = run_capability(FsWriteFile(), {"path": "out/x.txt", "content": "data"}, ctx, ctx.granted)
        assert out["bytes_written"] == 4
        got = run_capability(FsReadFile(), {"path": "out/x.txt"}, ctx, ctx.granted)
        assert got["content"] == "data"
        listed = run_capability(FsListDir(), {"path": "."}, ctx, ctx.granted)["entries"]
        assert any(e["name"] == "out" and e["is_dir"] for e in listed)

    def test_authority_denied_without_grant(self, store, workspace: Path) -> None:
        ctx = _ctx(store, workspace, granted=Authority(fs_write=("elsewhere/",)))
        with pytest.raises(AuthorityDenied):
            run_capability(FsWriteFile(), {"path": "here.txt", "content": "x"}, ctx, ctx.granted)


class TestRunTests:
    def test_real_pytest_pass_and_fail(self, store, workspace: Path) -> None:
        tests = workspace / "tests"
        tests.mkdir()
        (tests / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
        (tests / "test_bad.py").write_text("def test_bad():\n    assert 1 == 2\n")
        (workspace / "pytest.ini").write_text("[pytest]\n")
        ctx = _ctx(store, workspace)
        ok = run_capability(RepoRunTests(), {"cwd": ".", "args": ["-q", "tests/test_ok.py"]}, ctx, ctx.granted)
        assert ok["passed"] is True
        bad = run_capability(RepoRunTests(), {"cwd": ".", "args": ["-q", "tests/test_bad.py"]}, ctx, ctx.granted)
        assert bad["returncode"] != 0 and bad["passed"] is False
        assert "test_bad" in bad["stdout"]

    def test_probe_runs_real_pytest_version(self, store, workspace: Path) -> None:
        evidence = RepoRunTests().probe(_ctx(store, workspace))
        assert b"pytest" in evidence


class TestApplyPatch:
    def _diff(self, old: list[str], new: list[str]) -> str:
        import difflib

        return "".join(
            difflib.unified_diff(old, new, fromfile="a/mod.py", tofile="b/mod.py", lineterm="\n")
        ) + "\n"

    def test_apply_and_reject(self, store, workspace: Path) -> None:
        (workspace / "mod.py").write_text("value = 1\nprint(value)\n")
        ctx = _ctx(store, workspace)
        diff = self._diff(["value = 1\n", "print(value)\n"], ["value = 2\n", "print(value)\n"])
        out = run_capability(RepoApplyPatch(), {"cwd": ".", "diff": diff}, ctx, ctx.granted)
        assert out["applied"] == 1
        assert "value = 2" in (workspace / "mod.py").read_text()

        bad_diff = self._diff(["value = 999\n"], ["value = 1000\n"])
        before = (workspace / "mod.py").read_text()
        with pytest.raises(PatchError):
            run_capability(RepoApplyPatch(), {"cwd": ".", "diff": bad_diff}, ctx, ctx.granted)
        assert (workspace / "mod.py").read_text() == before


class TestSearchAndSummarize:
    def test_search_corpus_uses_fts(self, store, workspace: Path) -> None:
        text = "the secret number is forty two"
        store.index_chunk(
            {
                "chunk_id": "d:0",
                "doc_id": "d",
                "text": text,
                "ordinal": 0,
                "start": 0,
                "end": len(text),
                "sha": content_hash(text),
            }
        )
        ctx = _ctx(store, workspace)
        hits = run_capability(TextSearchCorpus(), {"query": "secret number", "k": 3}, ctx, ctx.granted)
        assert hits["hits"] and hits["hits"][0]["doc_id"] == "d"

    def test_summarize_consumes_recording(self, store, workspace: Path) -> None:
        ctx = _ctx(store, workspace)
        out = run_capability(
            TextSummarize(),
            {"text": "long text " * 500, "max_words": 5},
            ctx,
            ctx.granted,
        )
        assert out["summary"] == "SUMMARY TEXT"


class TestToolCallJournal:
    def test_events_wrap_invocation(self, store, workspace: Path) -> None:
        store.create_run("r_test", problem_sha="p")
        ctx = _ctx(store, workspace)
        # Read a file inside the workspace: `**` is workspace-relative, and a
        # journalling test should not depend on reading its own source.
        (workspace / "subject.txt").write_text("payload\n", encoding="utf-8")
        run_capability(FsReadFile(), {"path": "subject.txt"}, ctx, ctx.granted)
        kinds = [e.kind for e in store.events(run_id="r_test") if e.kind.startswith("tool_call")]
        assert kinds == ["tool_call_started", "tool_call_finished"]
        fin = [e for e in store.events(run_id="r_test") if e.kind == "tool_call_finished"][0]
        assert fin.payload["ok"] is True and fin.payload["output_sha"]


class TestResolveInputs:
    def test_template_binding(self, store, workspace: Path) -> None:
        bound = resolve_inputs({"a": "{{ x + 1 }}", "b": "literal"}, {"x": 41})
        assert bound == {"a": 42, "b": "literal"}


class TestRecordedChannelIsReplayNotFifo:
    """A recording must be bound to the REQUEST that produced it.

    Keying only on `session` made RecordedChannel a positional tape: reorder the
    plan, or ask something that was never recorded, and the next queued answer
    came back wrong-but-plausible with no error at all.
    """

    def _turn(self, prompt: str, answer: str) -> dict:
        return {"text": answer, "match": {"messages": [{"role": "user", "content": prompt}]}}

    def test_request_keyed_entry_replays_when_request_matches(self) -> None:
        ch = RecordedChannel({"llm": [self._turn("what is 2+2?", "four")]})
        got = ch.complete([{"role": "user", "content": "what is 2+2?"}], session="llm")
        assert got.text == "four"

    def test_unrecorded_request_fails_loudly_instead_of_answering(self) -> None:
        ch = RecordedChannel({"llm": [self._turn("what is 2+2?", "four")]})
        with pytest.raises(RecordingMismatch) as exc:
            ch.complete([{"role": "user", "content": "what is the capital of France?"}],
                        session="llm")
        detail = str(exc.value)
        assert "llm" in detail and "capital of France" in detail and "2+2" in detail

    def test_reordered_plan_does_not_get_the_other_answer(self) -> None:
        ch = RecordedChannel({"llm": [self._turn("first?", "A"), self._turn("second?", "B")]})
        with pytest.raises(RecordingMismatch):
            ch.complete([{"role": "user", "content": "second?"}], session="llm")

    def test_sampling_parameters_are_part_of_the_key(self) -> None:
        ch = RecordedChannel({"llm": [{"text": "hot", "match": {"temperature": 0.9,
                                                                "max_tokens": 32}}]})
        assert ch.complete([], session="llm", temperature=0.9, max_tokens=32).text == "hot"
        ch2 = RecordedChannel({"llm": [{"text": "hot", "match": {"temperature": 0.9}}]})
        with pytest.raises(RecordingMismatch, match="temperature"):
            ch2.complete([], session="llm", temperature=0.2)

    def test_legacy_unkeyed_tape_still_replays_in_order(self) -> None:
        """Compatibility: `RecordedChannel({"s": ["a", "b"]})` must keep working."""
        ch = RecordedChannel({"s": ["a", "b"]})
        assert [ch.complete([], session="s").text for _ in range(2)] == ["a", "b"]
        with pytest.raises(RecordingExhausted):
            ch.complete([], session="s")

    def test_malformed_recording_entry_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="text"):
            RecordedChannel({"s": [{"answer": "oops"}]})
        with pytest.raises(ValueError, match="nonsense"):
            RecordedChannel({"s": [{"text": "x", "match": {"nonsense": 1}}]})

    def test_empty_recording_set_is_not_an_echo_channel(self) -> None:
        """`{}` means "recorded, nothing recorded" -- distinguishable from "unconfigured"."""
        ch = make_channel("recorded", {})
        assert isinstance(ch, RecordedChannel), type(ch)
        with pytest.raises(RecordingExhausted):
            ch.complete([], session="anything")
        assert isinstance(make_channel("recorded", None), EchoChannel)


def _git(workspace: Path, *args: str) -> str:
    """Run a REAL git command in *workspace* and return stdout."""
    proc = subprocess.run(
        ["git", "-c", "user.email=sherpa@test", "-c", "user.name=sherpa", *args],
        cwd=workspace, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
    return proc.stdout


class TestApplyPatchMultiFile:
    """`_parse_unified_diff` flushed a pending hunk only at the NEXT `@@` or EOF.

    By then ``current_file`` had already advanced to the next `+++` line, so
    file N's hunks were filed under file N+1. When the two files share context
    lines the misfiling applies one file's content to another with no error at
    all -- silent data corruption.
    """

    def test_two_file_unified_diff_hits_both_files(self, store, workspace: Path) -> None:
        (workspace / "f1.txt").write_text("one\n", encoding="utf-8")
        (workspace / "f2.txt").write_text("two\n", encoding="utf-8")
        diff = (
            "--- a/f1.txt\n+++ b/f1.txt\n@@ -1,1 +1,1 @@\n-one\n+ONE\n"
            "--- a/f2.txt\n+++ b/f2.txt\n@@ -1,1 +1,1 @@\n-two\n+TWO\n"
        )
        parsed = _parse_unified_diff(diff)
        assert sorted(parsed) == ["f1.txt", "f2.txt"], (
            f"hunks were misattributed across the file boundary: {sorted(parsed)}"
        )
        ctx = _ctx(store, workspace)
        out = run_capability(RepoApplyPatch(), {"cwd": ".", "diff": diff}, ctx, ctx.granted)
        assert out["applied"] == 2
        assert (workspace / "f1.txt").read_text() == "ONE\n"
        assert (workspace / "f2.txt").read_text() == "TWO\n"

    def test_identical_context_lines_do_not_cross_contaminate(
        self, store, workspace: Path
    ) -> None:
        """The SILENT corruption case: two identical files, hunks at different lines.

        Misattribution files alpha's hunk under beta.py; because beta.py has the
        same content, alpha's hunk applies there cleanly. The old parser wrote
        BOTH edits into beta.py, left alpha.py untouched, and reported success.
        """
        shared = "header\nfirst = 0\nmiddle\nsecond = 0\n"
        (workspace / "alpha.py").write_text(shared, encoding="utf-8")
        (workspace / "beta.py").write_text(shared, encoding="utf-8")
        diff = (
            "--- a/alpha.py\n+++ b/alpha.py\n@@ -1,2 +1,2 @@\n"
            " header\n-first = 0\n+first = 111\n"
            "--- a/beta.py\n+++ b/beta.py\n@@ -3,2 +3,2 @@\n"
            " middle\n-second = 0\n+second = 222\n"
        )
        ctx = _ctx(store, workspace)
        out = run_capability(RepoApplyPatch(), {"cwd": ".", "diff": diff}, ctx, ctx.granted)
        assert out["applied"] == 2, f"only one file was touched: {out!r}"
        assert (workspace / "alpha.py").read_text() == (
            "header\nfirst = 111\nmiddle\nsecond = 0\n"
        ), "alpha.py was left untouched: its hunk was filed under beta.py"
        assert (workspace / "beta.py").read_text() == (
            "header\nfirst = 0\nmiddle\nsecond = 222\n"
        ), "beta.py absorbed alpha.py's edit as well as its own"

    def test_real_git_diff_of_two_modified_files_applies(
        self, store, workspace: Path
    ) -> None:
        """Generated by actually running `git diff` -- not hand-written."""
        repo = workspace / "repo"
        repo.mkdir()
        _git(repo, "init", "-q", ".")
        (repo / "f1.txt").write_text("one\nkeep\n", encoding="utf-8")
        (repo / "f2.txt").write_text("two\nkeep\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "init")
        (repo / "f1.txt").write_text("ONE\nkeep\n", encoding="utf-8")
        (repo / "f2.txt").write_text("TWO\nkeep\n", encoding="utf-8")
        diff = _git(repo, "diff")
        assert diff.count("diff --git") == 2, diff
        _git(repo, "checkout", "--", ".")
        assert (repo / "f1.txt").read_text() == "one\nkeep\n"

        ctx = _ctx(store, workspace)
        out = run_capability(RepoApplyPatch(), {"cwd": "repo", "diff": diff}, ctx, ctx.granted)
        assert out["applied"] == 2, out
        assert (repo / "f1.txt").read_text() == "ONE\nkeep\n"
        assert (repo / "f2.txt").read_text() == "TWO\nkeep\n"

    def test_real_git_diff_can_create_a_new_file(self, store, workspace: Path) -> None:
        """`@@ -0,0 +1,N @@` -- start=0 used to become idx=-1 and always raise."""
        repo = workspace / "repo"
        repo.mkdir()
        _git(repo, "init", "-q", ".")
        (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "init")
        (repo / "created.txt").write_text("alpha\nbeta\n", encoding="utf-8")
        _git(repo, "add", "created.txt")
        diff = _git(repo, "diff", "--cached")
        assert "@@ -0,0 +1,2 @@" in diff, diff
        _git(repo, "reset", "-q")
        (repo / "created.txt").unlink()

        ctx = _ctx(store, workspace)
        out = run_capability(RepoApplyPatch(), {"cwd": "repo", "diff": diff}, ctx, ctx.granted)
        assert out["applied"] == 1, out
        assert (repo / "created.txt").read_text() == "alpha\nbeta\n"

    def test_context_mismatch_still_rejected_loudly(self, store, workspace: Path) -> None:
        """The parser fix must not weaken the mismatch guard."""
        (workspace / "f1.txt").write_text("actual\n", encoding="utf-8")
        diff = "--- a/f1.txt\n+++ b/f1.txt\n@@ -1,1 +1,1 @@\n-expected\n+patched\n"
        ctx = _ctx(store, workspace)
        with pytest.raises(PatchError, match="context mismatch"):
            run_capability(RepoApplyPatch(), {"cwd": ".", "diff": diff}, ctx, ctx.granted)
        assert (workspace / "f1.txt").read_text() == "actual\n"

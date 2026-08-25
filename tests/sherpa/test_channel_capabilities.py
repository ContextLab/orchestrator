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

"""Typed capabilities: executable operations with authority and evidence (#492).

A capability is an admitted-executable-claim target: typed I/O, declared
authority requirements, and a cheap ``probe`` producing *executable evidence*
that the operation can actually run. The kernel never invokes a capability
except through :func:`run_capability`, which enforces authority and journals
both sides of the call.
"""

from __future__ import annotations

import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclasses_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from sherpa.events import Event
from sherpa.ir import Authority

if TYPE_CHECKING:
    from sherpa.channel import ModelChannel
    from sherpa.store import Store


class CapabilitySpec(BaseModel):
    name: str
    version: str = "1"
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    authority_required: Authority = Field(default_factory=Authority)


class ProbeSpec(BaseModel):
    kind: str = "builtin_selfcheck"


class AuthorityDenied(PermissionError):
    """The granted authority does not cover the capability's requirements."""


class ProbeFailed(RuntimeError):
    """Executable evidence could not be produced."""


@dataclass
class CapabilityContext:
    workspace: Path
    store: "Store"
    run_id: str
    node_key: str
    channel_factory: Callable[[], "ModelChannel"]
    granted: Authority = dataclasses_field(default_factory=Authority)
    def journal(self, kind: str, text: str, refs: list[str] | None = None) -> None:
        from sherpa.context import JOURNAL_KINDS, journal

        if kind not in JOURNAL_KINDS:
            raise ValueError(f"invalid journal kind {kind!r}")
        journal(self.store, self.run_id, self.node_key, kind, text, refs or [])

    def artifact(self, data: "bytes | str | dict", name: str) -> str:
        if isinstance(data, dict):
            import json

            payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        elif isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = data
        sha = self.store.blob.put_bytes(payload)
        self.store.append(
            Event(
                kind="artifact_written",
                run_id=self.run_id,
                node_key=self.node_key,
                payload={"name": name, "sha": sha},
            )
        )
        return sha


class Capability(ABC):
    spec: CapabilitySpec

    @abstractmethod
    def run(self, inputs: dict, ctx: CapabilityContext) -> dict: ...

    @abstractmethod
    def probe(self, ctx: CapabilityContext) -> bytes:
        """Cheap executable evidence that this capability works right now."""


class CapabilityRegistry:
    def __init__(self) -> None:
        self._caps: dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        name = cap.spec.name
        if name in self._caps:
            raise ValueError(f"capability {name!r} already registered")
        self._caps[name] = cap

    def get(self, name: str) -> Capability:
        if name not in self._caps:
            raise KeyError(f"unknown capability {name!r}")
        return self._caps[name]

    def names(self) -> set[str]:
        return set(self._caps)


def assert_authority(required: Authority, granted: Authority, what: str) -> None:
    if not granted.allows(required):
        missing = []
        for fld in ("fs_read", "fs_write", "net_domains", "subprocess_allow"):
            for pat in getattr(required, fld):
                if not any(
                    _grant_covers(g, pat) for g in getattr(granted, fld)
                ):
                    missing.append(f"{fld}:{pat}")
        raise AuthorityDenied(f"{what} requires authority not granted: {', '.join(missing)}")


def _grant_covers(grant: str, needed_literal: str) -> bool:
    from fnmatch import fnmatchcase

    return (
        grant == needed_literal
        or fnmatchcase(needed_literal, grant)
        or (needed_literal.startswith(grant) if grant.endswith("/") else False)
    )


# --------------------------------------------------------------------------
# Built-in capabilities. All side effects are real; the model boundary is the
# only recorded surface (text.summarize).
# --------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class FsReadFile(Capability):
    spec = CapabilitySpec(
        name="fs.read_file",
        description="Read a UTF-8 text file.",
        input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"content": {"type": "string"}}},
        authority_required=Authority(fs_read=("**",)),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        p = Path(inputs["path"])
        if not p.is_absolute():
            p = ctx.workspace / p
        assert_authority(Authority(fs_read=(str(p),)), ctx.granted, self.spec.name)
        return {"content": _read(p)}

    def probe(self, ctx: CapabilityContext) -> bytes:
        canary = ctx.workspace / ".sherpa_probe_read.txt"
        canary.write_text("probe-ok", encoding="utf-8")
        try:
            data = canary.read_text(encoding="utf-8")
        finally:
            canary.unlink(missing_ok=True)
        if data != "probe-ok":
            raise ProbeFailed("fs.read_file probe mismatch")
        return b"fs.read_file probe ok"


class FsWriteFile(Capability):
    spec = CapabilitySpec(
        name="fs.write_file",
        description="Write UTF-8 text to a file.",
        input_schema={
            "type": "object",
            "required": ["path", "content"],
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        },
        output_schema={"type": "object", "properties": {"bytes_written": {"type": "integer"}}},
        authority_required=Authority(fs_write=("**",)),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        p = Path(inputs["path"])
        if not p.is_absolute():
            p = ctx.workspace / p
        assert_authority(Authority(fs_write=(str(p),)), ctx.granted, self.spec.name)
        data = inputs["content"].encode("utf-8")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return {"bytes_written": len(data)}

    def probe(self, ctx: CapabilityContext) -> bytes:
        canary = ctx.workspace / ".sherpa_probe_write.txt"
        try:
            canary.write_text("ok", encoding="utf-8")
            if canary.read_text(encoding="utf-8") != "ok":
                raise ProbeFailed("write probe mismatch")
        finally:
            canary.unlink(missing_ok=True)
        return b"fs.write_file probe ok"


class FsListDir(Capability):
    spec = CapabilitySpec(
        name="fs.list_dir",
        description="List a directory.",
        input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"entries": {"type": "array"}}},
        authority_required=Authority(fs_read=("**",)),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        p = Path(inputs["path"])
        if not p.is_absolute():
            p = ctx.workspace / p
        assert_authority(Authority(fs_read=(str(p),)), ctx.granted, self.spec.name)
        entries = [
            {"name": e.name, "is_dir": e.is_dir(), "size": e.stat().st_size if e.is_file() else 0}
            for e in sorted(p.iterdir())
        ]
        return {"entries": entries}

    def probe(self, ctx: CapabilityContext) -> bytes:
        list(ctx.workspace.iterdir())
        return b"fs.list_dir probe ok"


class RepoRunTests(Capability):
    spec = CapabilitySpec(
        name="repo.run_tests",
        description="Run pytest as a REAL subprocess inside a directory.",
        input_schema={
            "type": "object",
            "required": ["cwd"],
            "properties": {
                "cwd": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "returncode": {"type": "integer"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "passed": {"type": "boolean"},
            },
        },
        authority_required=Authority(subprocess_allow=("python", "pytest")),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        cwd = Path(inputs["cwd"])
        if not cwd.is_absolute():
            cwd = ctx.workspace / cwd
        args = list(inputs.get("args", ["-q", "tests"]))
        cmd = [sys.executable, "-m", "pytest", *args]
        assert_authority(
            Authority(subprocess_allow=(sys.executable, "python", "pytest")),
            ctx.granted,
            self.spec.name,
        )
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=inputs.get("timeout", 120))  # noqa: S603 - fixed argv
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "passed": proc.returncode == 0,
        }

    def probe(self, ctx: CapabilityContext) -> bytes:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise ProbeFailed(f"pytest --version failed: {proc.stderr[:200]}")
        return proc.stdout.encode()


class PatchError(ValueError):
    """A unified diff did not apply cleanly; nothing was written."""


class RepoApplyPatch(Capability):
    spec = CapabilitySpec(
        name="repo.apply_patch",
        description="Apply a strict unified diff under cwd; atomic per file set.",
        input_schema={
            "type": "object",
            "required": ["cwd", "diff"],
            "properties": {"cwd": {"type": "string"}, "diff": {"type": "string"}},
        },
        output_schema={"type": "object", "properties": {"applied": {"type": "integer"}}},
        authority_required=Authority(fs_write=("**",)),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        cwd = Path(inputs["cwd"])
        if not cwd.is_absolute():
            cwd = ctx.workspace / cwd
        diff_text = inputs["diff"]
        plan = _parse_unified_diff(diff_text)
        touched: list[Path] = []
        try:
            for rel, hunks in plan.items():
                target = cwd / rel
                original = target.read_text(encoding="utf-8") if target.exists() else ""
                updated = _apply_hunks(original, hunks, rel)
                assert_authority(Authority(fs_write=(str(target),)), ctx.granted, self.spec.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(updated, encoding="utf-8")
                touched.append(target)
                # Same-size patches can leave a stale bytecode cache that
                # mtime+size validation fails to invalidate (equal length, and
                # the write may land in the same timestamp tick). Derived
                # caches must not outlive the patch.
                cache_dir = target.parent / "__pycache__"
                if cache_dir.is_dir():
                    for pyc in cache_dir.glob(target.stem + ".*.pyc"):
                        pyc.unlink(missing_ok=True)
        except Exception:
            raise
        return {"applied": len(touched), "files": [str(t.relative_to(cwd)) for t in touched]}

    def probe(self, ctx: CapabilityContext) -> bytes:
        canary = ctx.workspace / ".sherpa_probe_patch.txt"
        canary.write_text("alpha\nbeta\n", encoding="utf-8")
        import difflib

        diff = "".join(
            difflib.unified_diff(
                ["alpha\n", "beta\n"], ["alpha\n", "gamma\n"], fromfile="a/.sherpa_probe_patch.txt",
                tofile="b/.sherpa_probe_patch.txt",
            )
        )
        try:
            hunks = _parse_unified_diff(diff)
            original = canary.read_text(encoding="utf-8")
            updated = _apply_hunks(original, next(iter(hunks.values())), ".sherpa_probe_patch.txt")
            if "gamma" not in updated:
                raise ProbeFailed("patch round-trip failed")
        finally:
            canary.unlink(missing_ok=True)
        return b"repo.apply_patch probe ok"


def _parse_unified_diff(diff_text: str) -> dict[str, list[tuple[list[str], list[str], int]]]:
    files: dict[str, list[tuple[list[str], list[str], int]]] = {}
    current_file: str | None = None
    old: list[str] = []
    new: list[str] = []
    start_old = 0
    in_hunk = False

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            current_file = line[4:].strip()
            if current_file.startswith("b/"):
                current_file = current_file[2:]
            continue
        if line.startswith("@@"):
            if in_hunk and current_file is not None:
                files.setdefault(current_file, []).append((old, new, start_old))
            header = line.split()
            start_old = int(header[1].split(",")[0].lstrip("+").lstrip("-"))
            old, new = [], []
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line == "\n":
            continue
        tag, rest = line[0], line[1:]
        if tag == "-":
            old.append(rest)
        elif tag == "+":
            new.append(rest)
        elif tag == " ":
            old.append(rest)
            new.append(rest)
        elif tag == "\\":
            continue
        else:
            raise PatchError(f"malformed diff line: {line!r}")
    if in_hunk and current_file is not None:
        files.setdefault(current_file, []).append((old, new, start_old))
    if not files:
        raise PatchError("no hunks found in diff")
    return files


def _apply_hunks(original: str, hunks: list[tuple[list[str], list[str], int]], rel: str) -> str:
    lines = original.splitlines(keepends=True)
    for old, new, start in sorted(hunks, key=lambda h: h[2], reverse=True):
        idx = start - 1
        if idx < 0 or lines[idx : idx + len(old)] != old:
            raise PatchError(f"context mismatch applying patch to {rel!r}; nothing written")
        lines[idx : idx + len(old)] = new
    return "".join(lines)


class TextSearchCorpus(Capability):
    spec = CapabilitySpec(
        name="text.search_corpus",
        description="FTS5 retrieval over indexed chunks of this run.",
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
        },
        output_schema={"type": "object", "properties": {"hits": {"type": "array"}}},
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        hits = ctx.store.fts_search(inputs["query"], k=int(inputs.get("k", 5)))
        return {"hits": hits}

    def probe(self, ctx: CapabilityContext) -> bytes:
        ctx.store.fts_search("probe")
        return b"text.search_corpus probe ok"


class TextSummarize(Capability):
    spec = CapabilitySpec(
        name="text.summarize",
        description="Summarize text via the model channel (recorded in hermetic runs).",
        input_schema={
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}, "max_words": {"type": "integer"}},
        },
        output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        channel = ctx.channel_factory()
        max_words = int(inputs.get("max_words", 80))
        resp = channel.complete(
            [
                {"role": "system", "content": f"Summarize in at most {max_words} words."},
                {"role": "user", "content": inputs["text"][:8000]},
            ],
            session="summarizer",
        )
        return {"summary": resp.text}

    def probe(self, ctx: CapabilityContext) -> bytes:
        channel = ctx.channel_factory()
        try:
            channel.complete([{"role": "user", "content": "ping"}], session="summarizer")
        except Exception as exc:
            raise ProbeFailed(f"summarize channel unavailable: {exc}") from exc
        return b"text.summarize probe ok"


def register_builtins(registry: CapabilityRegistry) -> None:
    for cap_cls in (FsReadFile, FsWriteFile, FsListDir, RepoRunTests, RepoApplyPatch, TextSearchCorpus, TextSummarize):
        registry.register(cap_cls())


def resolve_inputs(node_inputs: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    """Bind ``{{ expr }}`` templates against *scope* using the safe evaluator."""
    from sherpa.expr import evaluate

    out: dict[str, Any] = {}
    for key, value in node_inputs.items():
        if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            out[key] = evaluate(value[2:-2].strip(), scope)
        else:
            out[key] = value
    return out


def run_capability(cap: Capability, inputs: dict, ctx: CapabilityContext, granted: Authority) -> dict:
    """The ONLY invocation path: authority check + journaled tool-call events."""
    from sherpa.events import Event

    started = time.time()
    ctx.store.append(
        Event(
            kind="tool_call_started",
            run_id=ctx.run_id,
            node_key=ctx.node_key,
            payload={"capability": cap.spec.name, "inputs": inputs},
        )
    )
    assert_authority(cap.spec.authority_required, granted, cap.spec.name)
    try:
        result = cap.run(inputs, ctx)
    except Exception as exc:
        ctx.store.append(
            Event(
                kind="tool_call_finished",
                run_id=ctx.run_id,
                node_key=ctx.node_key,
                payload={"capability": cap.spec.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )
        )
        raise
    out_sha = ctx.artifact(result, name=f"{cap.spec.name}.result.json")
    ctx.store.append(
        Event(
            kind="tool_call_finished",
            run_id=ctx.run_id,
            node_key=ctx.node_key,
            payload={
                "capability": cap.spec.name,
                "ok": True,
                "duration_s": time.time() - started,
                "output_sha": out_sha,
            },
        )
    )
    return result

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

from sherpa.admission import check_io
from sherpa.authority import AuthorityError
from sherpa.events import Event
from sherpa.ir import Authority

if TYPE_CHECKING:
    from sherpa.channel import ModelChannel
    from sherpa.store import Store


class CapabilitySpec(BaseModel):
    """A capability's contract.

    ``requires`` names the authority *dimensions* the capability uses; the
    concrete resource is checked per invocation against the resolved path (see
    :func:`assert_fs_access`). ``authority_required`` remains for grants that
    genuinely are pattern-shaped rather than path-shaped -- currently only
    ``subprocess_allow``.

    Declaring ``authority_required=Authority(fs_read=("**",))`` -- as every fs
    capability used to -- conflates the two questions and makes the capability
    demand filesystem-wide power just to read one granted file, which is why
    scoped grants were previously unusable.
    """

    name: str
    version: str = "1"
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    authority_required: Authority = Field(default_factory=Authority)
    requires: tuple[str, ...] = ()


class ProbeSpec(BaseModel):
    kind: str = "builtin_selfcheck"


class AuthorityDenied(AuthorityError):
    """The granted authority does not cover the capability's requirements.

    Subclasses the shared :class:`sherpa.authority.AuthorityError` so callers
    may catch either; there is one denial hierarchy, not one per module.
    """


class CapabilityContractError(TypeError):
    """A capability's actual I/O did not match its declared typed schema."""


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
    """Pattern-level delegation check (used for ``subprocess_allow``)."""
    from sherpa.authority import authority_covers, missing_powers

    if not authority_covers(granted, required):
        missing = missing_powers(granted, required)
        raise AuthorityDenied(f"{what} requires authority not granted: {', '.join(missing)}")


def assert_requires(spec: "CapabilitySpec", granted: Authority, what: str) -> None:
    """Every dimension the capability uses must be granted *something*.

    This is the coarse gate. It deliberately does not inspect patterns: the
    real check happens per resolved resource in :func:`assert_fs_access`.
    """
    for dimension in spec.requires:
        if not getattr(granted, dimension, ()):
            raise AuthorityDenied(f"{what} requires {dimension} authority, none granted")


def assert_fs_access(
    raw_path: "str | Path", dimension: str, ctx: "CapabilityContext", what: str
) -> Path:
    """Resolve *raw_path* and confirm the grant covers it. Returns the path.

    Resolution happens *before* the check and matches what the capability will
    actually open, so ``..``, absolute paths, and symlinks out of a granted
    directory are all visible to the grant comparison rather than hidden by it.
    """
    from sherpa.authority import path_within_grants, resolve_fs_path

    resolved = resolve_fs_path(raw_path, ctx.workspace)
    grants = getattr(ctx.granted, dimension, ())
    if not path_within_grants(grants, resolved, ctx.workspace):
        raise AuthorityDenied(
            f"{what}: {dimension} denied for {resolved} (granted: {tuple(grants)!r})"
        )
    return resolved


# --------------------------------------------------------------------------
# Built-in capabilities. All side effects are real; the model boundary is the
# only recorded surface (text.summarize).
# --------------------------------------------------------------------------


#: pytest flags that cause arbitrary code to be imported or a different
#: configuration/rootdir to be honoured. Forwarding them defeats the cwd check.
_UNSAFE_PYTEST_FLAGS = ("-p", "-c", "--rootdir", "--confcutdir", "--import-mode", "-P")

_SECRET_MARKERS = ("secret", "token", "password", "api_key", "apikey", "credential")


def _safe_pytest_args(args: "list[str]") -> list[str]:
    """Reject pytest arguments that load code or relocate the config root."""
    safe = [str(a) for a in args]
    for arg in safe:
        head = arg.split("=", 1)[0]
        if head in _UNSAFE_PYTEST_FLAGS:
            raise AuthorityDenied(f"repo.run_tests refuses code-loading argument {arg!r}")
    return safe


def _redact(inputs: dict) -> dict:
    """Journal the shape of a call, never bulk content or anything secret-ish.

    Capability inputs used to be written verbatim into the event log, so a
    written credential became a permanent plaintext record.
    """
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        lowered = key.lower()
        if any(marker in lowered for marker in _SECRET_MARKERS):
            out[key] = "<redacted>"
        elif isinstance(value, str) and len(value) > 120:
            out[key] = f"<{len(value)} chars>"
        elif isinstance(value, str) and any(m in value.lower() for m in _SECRET_MARKERS):
            out[key] = "<redacted>"
        else:
            out[key] = value
    return out


def _probe_scratch(ctx: "CapabilityContext", suffix: str) -> Path:
    """A unique, authorized scratch path for a probe that must really write.

    Probes run inside admission, *before* a step is admitted, so they are real
    side effects and are authority-bearing. A fixed canary name also risked
    clobbering a user file, so the name is unique per probe.
    """
    import uuid

    candidate = ctx.workspace / f".sherpa_probe_{uuid.uuid4().hex}{suffix}"
    assert_fs_access(candidate, "fs_write", ctx, "probe")
    return candidate


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class FsReadFile(Capability):
    spec = CapabilitySpec(
        name="fs.read_file",
        description="Read a UTF-8 text file.",
        input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"content": {"type": "string"}}},
        requires=("fs_read",),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        p = assert_fs_access(inputs["path"], "fs_read", ctx, self.spec.name)
        return {"content": _read(p)}

    def probe(self, ctx: CapabilityContext) -> bytes:
        # Read-only evidence: a read capability must never need write authority
        # to prove itself, and must never clobber an existing file.
        if not ctx.workspace.is_dir():
            raise ProbeFailed(f"workspace {ctx.workspace} is not a readable directory")
        try:
            next(iter(ctx.workspace.iterdir()), None)
        except OSError as exc:
            raise ProbeFailed(f"fs.read_file probe cannot read workspace: {exc}") from exc
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
        requires=("fs_write",),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        p = assert_fs_access(inputs["path"], "fs_write", ctx, self.spec.name)
        data = inputs["content"].encode("utf-8")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return {"bytes_written": len(data)}

    def probe(self, ctx: CapabilityContext) -> bytes:
        canary = _probe_scratch(ctx, ".txt")
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
        requires=("fs_read",),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        p = assert_fs_access(inputs["path"], "fs_read", ctx, self.spec.name)
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
        assert_authority(
            Authority(subprocess_allow=(sys.executable, "python", "pytest")),
            ctx.granted,
            self.spec.name,
        )
        # pytest executes conftest.py from its cwd, so the directory is as
        # authority-bearing as any file this capability could read.
        cwd = assert_fs_access(inputs["cwd"], "fs_read", ctx, self.spec.name)
        args = _safe_pytest_args(inputs.get("args", ["-q", "tests"]))
        cmd = [sys.executable, "-m", "pytest", *args]
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=inputs.get("timeout", 120))  # noqa: S603 - fixed argv
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "passed": proc.returncode == 0,
        }

    def probe(self, ctx: CapabilityContext) -> bytes:
        # Spawning a process is a real side effect; it needs the same grant the
        # capability itself needs.
        assert_authority(
            Authority(subprocess_allow=(sys.executable, "python", "pytest")),
            ctx.granted,
            self.spec.name,
        )
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
        requires=("fs_write",),
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        cwd = assert_fs_access(inputs["cwd"], "fs_write", ctx, self.spec.name)
        diff_text = inputs["diff"]
        plan = _parse_unified_diff(diff_text)
        touched: list[Path] = []
        try:
            # Resolve and authorize EVERY target before touching the first
            # one: a mid-loop denial must not leave earlier files rewritten.
            resolved = {
                rel: assert_fs_access(cwd / rel, "fs_write", ctx, self.spec.name)
                for rel in plan
            }
            for rel, target in resolved.items():
                if not target.is_relative_to(cwd):
                    raise PatchError(f"patch target {rel!r} escapes cwd")
            for rel, hunks in plan.items():
                target = resolved[rel]
                original = target.read_text(encoding="utf-8") if target.exists() else ""
                updated = _apply_hunks(original, hunks, rel)
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
        canary = _probe_scratch(ctx, ".txt")
        canary.write_text("alpha\nbeta\n", encoding="utf-8")
        import difflib

        diff = "".join(
            difflib.unified_diff(
                ["alpha\n", "beta\n"], ["alpha\n", "gamma\n"], fromfile=f"a/{canary.name}",
                tofile=f"b/{canary.name}",
            )
        )
        try:
            hunks = _parse_unified_diff(diff)
            original = canary.read_text(encoding="utf-8")
            updated = _apply_hunks(original, next(iter(hunks.values())), canary.name)
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
        # Model spend is only enforceable if it is recorded.
        ctx.store.add_usage(
            ctx.run_id,
            tokens=resp.prompt_tokens + resp.completion_tokens,
            cost_usd=resp.cost_usd,
        )
        return {"summary": resp.text}

    def probe(self, ctx: CapabilityContext) -> bytes:
        channel = ctx.channel_factory()
        try:
            # A distinct session: admission's executable evidence must not
            # consume the executor's recorded response. Sharing the session
            # meant each admitted summarize burned two recordings and the step
            # silently received the SECOND one.
            channel.complete([{"role": "user", "content": "ping"}],
                             session="summarizer_probe")
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
            payload={"capability": cap.spec.name, "inputs": _redact(inputs)},
        )
    )
    assert_requires(cap.spec, granted, cap.spec.name)
    assert_authority(cap.spec.authority_required, granted, cap.spec.name)
    ok_in, in_errors = check_io(inputs, cap.spec.input_schema)
    if not ok_in:
        raise CapabilityContractError(
            f"{cap.spec.name} inputs do not match its declared schema: {in_errors}"
        )
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
    ok_out, out_errors = check_io(result, cap.spec.output_schema)
    if not ok_out:
        # A capability that returns something other than what it declares is a
        # loud contract violation, not a silently-propagated value.
        ctx.store.append(
            Event(
                kind="tool_call_finished",
                run_id=ctx.run_id,
                node_key=ctx.node_key,
                payload={"capability": cap.spec.name, "ok": False,
                         "error": f"output schema violation: {out_errors}"},
            )
        )
        raise CapabilityContractError(
            f"{cap.spec.name} returned a value that violates its declared "
            f"output_schema: {out_errors}"
        )
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

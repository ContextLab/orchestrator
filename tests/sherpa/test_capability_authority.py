"""Capabilities are the only code in sherpa that touches the filesystem, the
network, or a subprocess, so ``run_capability`` is *the* security boundary.

Each test here reproduces an escape demonstrated end-to-end against the pre-fix
tree with a realistic scoped grant. The original suite missed all of them
because every capability test granted ``**`` (so the per-path check never ran)
and every denial test used a grant form without a trailing ``*`` (the one shape
the old matcher did not mis-handle).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from sherpa.authority import AuthorityError
from sherpa.capabilities import (
    AuthorityDenied,
    CapabilityContext,
    CapabilityRegistry,
    register_builtins,
    run_capability,
)
from sherpa.ir import Authority
from sherpa.store import Store

DENIALS = (AuthorityDenied, AuthorityError, PermissionError)


@pytest.fixture()
def env(tmp_path: Path):
    """A workspace granted only ``src/**``, plus secrets living outside it."""
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "ok.txt").write_text("in-scope\n", encoding="utf-8")

    secret = tmp_path / "outside_secret.txt"
    secret.write_text("TOP-SECRET-OUTSIDE\n", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("ORIGINAL\n", encoding="utf-8")

    registry = CapabilityRegistry()
    register_builtins(registry)
    store = Store(ws / "s.db")
    store.create_run("run_captest", "problemsha")
    granted = Authority(fs_read=("src/**",), fs_write=("src/**",))
    ctx = CapabilityContext(
        workspace=ws,
        store=store,
        run_id="run_captest",
        node_key="n",
        channel_factory=lambda: None,
        granted=granted,
    )
    yield {
        "ws": ws,
        "secret": secret,
        "victim": victim,
        "registry": registry,
        "ctx": ctx,
        "granted": granted,
    }
    store.close()


def _run(env, capability: str, inputs: dict):
    return run_capability(env["registry"].get(capability), inputs, env["ctx"], env["granted"])


# --------------------------------------------------------------------------
# scoped grants must WORK -- otherwise the only usable grant is "everything"
# --------------------------------------------------------------------------


def test_scoped_grant_permits_in_scope_read(env) -> None:
    """Pre-fix, a scoped fs grant made every fs capability unusable, because the
    capability declared ``authority_required=fs_read=('**',)``."""
    assert _run(env, "fs.read_file", {"path": "src/ok.txt"})["content"] == "in-scope\n"


def test_scoped_grant_permits_in_scope_write(env) -> None:
    result = _run(env, "fs.write_file", {"path": "src/new.txt", "content": "hello"})
    assert result["bytes_written"] == 5
    assert (env["ws"] / "src" / "new.txt").read_text(encoding="utf-8") == "hello"


# --------------------------------------------------------------------------
# reads must not escape the grant
# --------------------------------------------------------------------------


def test_absolute_path_outside_grant_is_denied(env) -> None:
    with pytest.raises(DENIALS):
        _run(env, "fs.read_file", {"path": str(env["secret"])})
    assert env["secret"].read_text(encoding="utf-8") == "TOP-SECRET-OUTSIDE\n"


def test_dotdot_traversal_is_denied(env) -> None:
    escape = os.path.join("src", "..", "..", env["secret"].name)
    with pytest.raises(DENIALS):
        _run(env, "fs.read_file", {"path": escape})


def test_symlink_out_of_grant_is_denied(env) -> None:
    link = env["ws"] / "src" / "link.txt"
    os.symlink(env["secret"], link)
    with pytest.raises(DENIALS):
        _run(env, "fs.read_file", {"path": "src/link.txt"})


def test_read_outside_grant_but_inside_workspace_is_denied(env) -> None:
    """Containment is the *grant*, not merely the workspace."""
    (env["ws"] / "private").mkdir()
    (env["ws"] / "private" / "k.txt").write_text("nope\n", encoding="utf-8")
    with pytest.raises(DENIALS):
        _run(env, "fs.read_file", {"path": "private/k.txt"})


# --------------------------------------------------------------------------
# writes must not escape the grant
# --------------------------------------------------------------------------


def test_write_outside_grant_is_denied_and_leaves_target_untouched(env) -> None:
    with pytest.raises(DENIALS):
        _run(env, "fs.write_file", {"path": str(env["victim"]), "content": "PWNED"})
    assert env["victim"].read_text(encoding="utf-8") == "ORIGINAL\n"


def test_write_traversal_is_denied_and_creates_nothing(env) -> None:
    target = env["ws"].parent / "escaped_write.txt"
    with pytest.raises(DENIALS):
        _run(env, "fs.write_file", {"path": "src/../../escaped_write.txt", "content": "x"})
    assert not target.exists()


def test_read_only_grant_cannot_write(env) -> None:
    env["ctx"].granted = Authority(fs_read=("src/**",))
    env["granted"] = env["ctx"].granted
    with pytest.raises(DENIALS):
        _run(env, "fs.write_file", {"path": "src/nope.txt", "content": "x"})
    assert not (env["ws"] / "src" / "nope.txt").exists()


# --------------------------------------------------------------------------
# subprocess: cwd is an authority-bearing input, and args can load code
# --------------------------------------------------------------------------


def test_run_tests_denies_cwd_outside_granted_scope(env) -> None:
    """pytest executes ``conftest.py`` from its cwd, so an unchecked cwd is
    arbitrary code execution under ``subprocess_allow`` alone."""
    hostile = env["ws"].parent / "hostile"
    hostile.mkdir()
    marker = env["ws"].parent / "conftest_ran.txt"
    (hostile / "conftest.py").write_text(
        f"open({str(marker)!r}, 'w').write('executed')\n", encoding="utf-8"
    )
    env["ctx"].granted = Authority(subprocess_allow=("python", "pytest"))
    env["granted"] = env["ctx"].granted
    with pytest.raises(DENIALS):
        _run(env, "repo.run_tests", {"cwd": str(hostile)})
    assert not marker.exists(), "conftest.py executed outside granted authority"


def test_run_tests_rejects_code_loading_args(env) -> None:
    env["ctx"].granted = Authority(
        fs_read=("src/**",), fs_write=("src/**",), subprocess_allow=("python", "pytest")
    )
    env["granted"] = env["ctx"].granted
    for hostile_args in (["-p", "evil"], ["-c", "/tmp/evil.ini"], ["--rootdir", "/"]):
        with pytest.raises((AuthorityDenied, AuthorityError, PermissionError, ValueError)):
            _run(env, "repo.run_tests", {"cwd": "src", "args": hostile_args})


def test_run_tests_without_subprocess_grant_is_denied(env) -> None:
    with pytest.raises(DENIALS):
        _run(env, "repo.run_tests", {"cwd": "src"})


# --------------------------------------------------------------------------
# patch application must check containment BEFORE writing
# --------------------------------------------------------------------------


def test_apply_patch_absolute_target_writes_nothing(env) -> None:
    """Pre-fix, the file was written and *then* the containment check raised."""
    victim = env["victim"]
    diff = (
        f"--- a{victim}\n"
        f"+++ b{victim}\n"
        "@@ -1,1 +1,1 @@\n"
        "-ORIGINAL\n"
        "+OVERWRITTEN\n"
    )
    with pytest.raises((AuthorityDenied, AuthorityError, PermissionError, ValueError)):
        _run(env, "repo.apply_patch", {"cwd": "src", "diff": diff})
    assert victim.read_text(encoding="utf-8") == "ORIGINAL\n"


def test_apply_patch_traversal_target_writes_nothing(env) -> None:
    target = env["ws"].parent / "patched_escape.txt"
    diff = (
        "--- a/../../patched_escape.txt\n"
        "+++ b/../../patched_escape.txt\n"
        "@@ -0,0 +1,1 @@\n"
        "+pwned\n"
    )
    with pytest.raises((AuthorityDenied, AuthorityError, PermissionError, ValueError)):
        _run(env, "repo.apply_patch", {"cwd": "src", "diff": diff})
    assert not target.exists()


# --------------------------------------------------------------------------
# probes run real side effects, so they are authority-bearing too
# --------------------------------------------------------------------------


def test_probe_does_not_write_without_write_authority(env) -> None:
    """``admission`` runs ``cap.probe(ctx)`` before the step is admitted; the
    read probe wrote a canary into the workspace under a read-only grant."""
    env["ctx"].granted = Authority(fs_read=("src/**",))
    before = set(p.name for p in env["ws"].rglob("*"))
    cap = env["registry"].get("fs.read_file")
    try:
        cap.probe(env["ctx"])
    except DENIALS:
        pass
    after = set(p.name for p in env["ws"].rglob("*"))
    assert after == before, f"probe created {after - before} without write authority"


def test_probe_never_clobbers_an_existing_file(env) -> None:
    canary = env["ws"] / ".sherpa_probe_read.txt"
    canary.write_text("PRECIOUS", encoding="utf-8")
    cap = env["registry"].get("fs.read_file")
    try:
        cap.probe(env["ctx"])
    except Exception:
        pass
    assert not canary.exists() or canary.read_text(encoding="utf-8") == "PRECIOUS"


# --------------------------------------------------------------------------
# secrets must not be journaled verbatim
# --------------------------------------------------------------------------


def test_capability_inputs_are_not_journaled_verbatim(env) -> None:
    secret_value = "AWS_SECRET_ACCESS_KEY=abc123SECRET"
    _run(env, "fs.write_file", {"path": "src/creds.txt", "content": secret_value})
    blob = "".join(
        str(e.payload) for e in env["ctx"].store.events(run_id="run_captest")
    )
    assert secret_value not in blob, "raw capability inputs were written into the event log"

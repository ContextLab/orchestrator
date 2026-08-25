"""Authority containment is a security boundary (#492: "No plan can exceed
parent authority").

Every test here corresponds to an escape that was demonstrated against the
pre-fix tree. The grant forms used are the *realistic* ones (``src/**``,
``out/*``, absolute workspace subtrees) rather than the bare-prefix forms that
the original suite happened to pick, because a trailing ``*`` was exactly the
shape that defeated the old matcher.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sherpa.authority import (
    authority_covers,
    path_within_grants,
    resolve_fs_path,
)
from sherpa.ir import Authority


# --------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------


def test_relative_path_resolves_against_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    assert resolve_fs_path("src/a.txt", ws) == (ws / "src" / "a.txt").resolve()


def test_dotdot_is_collapsed_not_preserved(tmp_path: Path) -> None:
    """``src/../../escape`` must resolve OUTSIDE the workspace, so the
    containment check can see it. The old code compared the raw string."""
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    resolved = resolve_fs_path("src/../../escape.txt", ws)
    assert ".." not in resolved.parts
    assert not resolved.is_relative_to(ws.resolve())


def test_symlink_is_followed_to_its_real_target(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = ws / "src" / "link.txt"
    os.symlink(outside, link)
    assert resolve_fs_path(str(link), ws) == outside.resolve()


# --------------------------------------------------------------------------
# per-resource containment -- the gate that actually protects the filesystem
# --------------------------------------------------------------------------


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "out").mkdir()
    return root


@pytest.mark.parametrize("grant", ["src/**", "src/*", "src/", "src"])
def test_grant_covers_its_own_subtree(ws: Path, grant: str) -> None:
    target = resolve_fs_path("src/a.txt", ws)
    assert path_within_grants((grant,), target, ws) is True


@pytest.mark.parametrize(
    "grant", ["src/**", "src/*", "src/", "src", "out/*", "docs/**"]
)
@pytest.mark.parametrize(
    "escape",
    [
        "/etc/passwd",
        "../../etc/shadow",
        "src/../../escaped.txt",
        "~/.ssh/id_rsa",
    ],
)
def test_no_grant_ever_covers_a_path_outside_itself(
    ws: Path, grant: str, escape: str
) -> None:
    """The pre-fix matcher returned True for every one of these combinations,
    because its final clause reduced to ``candidate.startswith("")``."""
    target = resolve_fs_path(escape, ws)
    assert path_within_grants((grant,), target, ws) is False


def test_symlink_out_of_a_granted_directory_is_denied(ws: Path) -> None:
    outside = ws.parent / "outside_secret.txt"
    outside.write_text("TOP-SECRET", encoding="utf-8")
    os.symlink(outside, ws / "src" / "link.txt")
    target = resolve_fs_path("src/link.txt", ws)
    assert path_within_grants(("src/**",), target, ws) is False


def test_single_star_does_not_cross_a_directory_separator(ws: Path) -> None:
    """``src/*`` is one segment; ``src/**`` is any depth."""
    deep = resolve_fs_path("src/pkg/deep.txt", ws)
    assert path_within_grants(("src/*",), deep, ws) is False
    assert path_within_grants(("src/**",), deep, ws) is True


def test_sibling_prefix_is_not_covered(ws: Path) -> None:
    """``src`` must not cover ``src_secret`` -- a raw ``startswith`` would."""
    (ws / "src_secret").mkdir()
    target = resolve_fs_path("src_secret/k.txt", ws)
    assert path_within_grants(("src",), target, ws) is False


def test_empty_grant_list_denies_everything(ws: Path) -> None:
    assert path_within_grants((), resolve_fs_path("src/a.txt", ws), ws) is False


def test_empty_pattern_string_denies(ws: Path) -> None:
    assert path_within_grants(("",), resolve_fs_path("src/a.txt", ws), ws) is False


def test_absolute_grant_covers_only_its_own_subtree(ws: Path) -> None:
    grant = str(ws / "src") + "/**"
    assert path_within_grants((grant,), resolve_fs_path("src/a.txt", ws), ws) is True
    assert path_within_grants((grant,), resolve_fs_path("out/a.txt", ws), ws) is False


def test_grant_naming_a_single_file_covers_only_that_file(ws: Path) -> None:
    assert path_within_grants(("src/a.txt",), resolve_fs_path("src/a.txt", ws), ws) is True
    assert path_within_grants(("src/a.txt",), resolve_fs_path("src/b.txt", ws), ws) is False


# --------------------------------------------------------------------------
# delegation: a child may only ever narrow a parent
# --------------------------------------------------------------------------


def test_child_cannot_widen_a_wildcard_parent_grant() -> None:
    """The headline delegation bypass: with the old matcher this returned True
    because the parent grant ended in ``*``."""
    parent = Authority(fs_read=("src/**",), fs_write=("src/**",))
    child = Authority(fs_read=("**",), fs_write=("/etc/**", "~/.ssh/**"))
    assert authority_covers(parent, child) is False
    assert parent.allows(child) is False


def test_child_narrowing_is_permitted() -> None:
    parent = Authority(fs_read=("src/**",))
    child = Authority(fs_read=("src/pkg/**",))
    assert authority_covers(parent, child) is True
    assert parent.allows(child) is True


def test_identical_authority_is_covered() -> None:
    a = Authority(fs_read=("src/**",), subprocess_allow=("pytest",))
    assert authority_covers(a, a) is True


def test_empty_child_is_always_covered() -> None:
    assert authority_covers(Authority(), Authority()) is True
    assert authority_covers(Authority(fs_read=("src/**",)), Authority()) is True


def test_empty_parent_covers_nothing() -> None:
    assert authority_covers(Authority(), Authority(fs_read=("a.txt",))) is False


def test_widening_is_rejected_in_every_dimension() -> None:
    parent = Authority(
        fs_read=("src/**",),
        fs_write=("out/**",),
        net_domains=("example.com",),
        subprocess_allow=("pytest",),
    )
    for widened in (
        Authority(fs_read=("**",)),
        Authority(fs_write=("**",)),
        Authority(net_domains=("evil.net",)),
        Authority(subprocess_allow=("bash",)),
    ):
        assert authority_covers(parent, widened) is False, widened


def test_net_domain_wildcard_does_not_cover_unrelated_domain() -> None:
    parent = Authority(net_domains=("*.example.com",))
    assert authority_covers(parent, Authority(net_domains=("api.example.com",))) is True
    assert authority_covers(parent, Authority(net_domains=("evil.net",))) is False
    assert authority_covers(parent, Authority(net_domains=("example.com.evil.net",))) is False


def test_subprocess_grant_is_exact_not_prefix() -> None:
    parent = Authority(subprocess_allow=("python",))
    assert authority_covers(parent, Authority(subprocess_allow=("python",))) is True
    assert authority_covers(parent, Authority(subprocess_allow=("python-evil",))) is False


def test_delegation_is_transitive_and_never_widens() -> None:
    root = Authority(fs_read=("src/**",))
    mid = Authority(fs_read=("src/pkg/**",))
    leaf = Authority(fs_read=("src/pkg/deep/**",))
    assert authority_covers(root, mid) and authority_covers(mid, leaf)
    assert authority_covers(root, leaf)
    assert not authority_covers(leaf, mid)


# --------------------------------------------------------------------------
# what "**" means -- a deliberate, load-bearing choice
# --------------------------------------------------------------------------


def test_bare_double_star_is_workspace_relative_not_filesystem_wide(ws: Path) -> None:
    """``**`` grants everything *under the workspace*, not the whole disk.

    The pre-fix matcher treated any ``*``-suffixed grant as unlimited, so this
    distinction did not exist. Making ``**`` workspace-relative means the
    documented quickstart grant cannot reach ``/etc`` by accident; asking for
    the filesystem now requires saying so explicitly with ``/**``.
    """
    inside = resolve_fs_path("src/a.txt", ws)
    outside = Path("/etc/passwd")
    assert path_within_grants(("**",), inside, ws) is True
    assert path_within_grants(("**",), outside, ws) is False


def test_absolute_double_star_is_filesystem_wide(ws: Path) -> None:
    assert path_within_grants(("/**",), Path("/etc/passwd"), ws) is True
    assert path_within_grants(("/**",), resolve_fs_path("src/a.txt", ws), ws) is True


def test_workspace_relative_grant_cannot_be_widened_to_filesystem(ws: Path) -> None:
    parent = Authority(fs_read=("**",))
    assert authority_covers(parent, Authority(fs_read=("/**",))) is False
    assert authority_covers(Authority(fs_read=("/**",)), Authority(fs_read=("**",))) is True

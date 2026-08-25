"""The single authority implementation for sherpa (#492).

Before this module existed there were three divergent matchers — one in
``ir.Authority.allows``, one in ``capabilities.assert_authority``, and one in
``admission.assert_authority_granted`` — which answered the same question
differently. The ``ir`` one reduced to ``candidate.startswith("")`` for any
grant ending in ``*``, i.e. it allowed everything, and nothing anywhere
normalized a filesystem path, so ``..``, absolute paths, and symlinks escaped
every grant.

Two distinct questions are kept separate here, because conflating them is what
made the old model unusable:

1. **Delegation** — may a child plan hold this pattern set at all, given the
   parent's? Answered by :func:`authority_covers` over *patterns*.
2. **Access** — may this specific, fully-resolved resource be touched?
   Answered by :func:`path_within_grants` over a *resolved path*.

Both fail closed: an empty grant list, an empty pattern, or anything the rules
below do not explicitly admit is denied.

Grant syntax (POSIX-style, ``/``-separated):

``src``/``src/``   a literal prefix; covers that path and everything beneath it
``src/*``          exactly one segment beneath ``src``
``src/**``         any depth beneath ``src`` (including ``src`` itself)
``src/a.txt``      that one file only
``**``             everything beneath the *workspace* -- NOT the whole disk
``/**``            the entire filesystem; the only way to ask for it

Relative grants are anchored at the run's workspace, so the quickstart's
``{"fs_read": ["**"]}`` cannot reach ``/etc`` by accident. Reaching outside the
workspace has to be spelled out, either as ``/**`` or as an explicit absolute
subtree such as ``/srv/data/**``.

Non-filesystem dimensions (``net_domains``, ``subprocess_allow``) are matched as
whole tokens with :func:`pattern_covers`; they are not paths and are never
treated as prefixes.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sherpa.ir import Authority

#: Every dimension of delegated power, in a fixed order.
DIMENSIONS: tuple[str, ...] = ("fs_read", "fs_write", "net_domains", "subprocess_allow")

#: Dimensions whose grants denote filesystem paths.
FS_DIMENSIONS: tuple[str, ...] = ("fs_read", "fs_write")

_WILDCARD_CHARS = ("*", "?", "[")


class AuthorityError(PermissionError):
    """A resource or pattern was not covered by the granted authority."""


# --------------------------------------------------------------------------
# pattern parsing
# --------------------------------------------------------------------------


def _split_pattern(pattern: str) -> tuple[str, list[str]]:
    """Split *pattern* into (literal prefix, glob tail segments).

    ``"src/pkg/**"`` -> ``("src/pkg", ["**"])``; ``"src/a.txt"`` -> ``("src/a.txt", [])``.
    A leading ``/`` is preserved so absolute grants stay absolute.
    """
    cleaned = pattern.rstrip("/")
    if not cleaned:
        # "/" (root) or "" -- "" is rejected by callers; "/" keeps its slash.
        cleaned = pattern
    segments = cleaned.split("/")
    for i, segment in enumerate(segments):
        if any(ch in segment for ch in _WILDCARD_CHARS):
            prefix = "/".join(segments[:i])
            # An absolute pattern whose first wildcard is the first segment
            # ("/**") joins to "", which would silently re-root it at the
            # workspace. Keep it anchored at "/".
            if not prefix and cleaned.startswith("/"):
                prefix = "/"
            return prefix, segments[i:]
    return cleaned, []


def _glob_segments(patterns: Sequence[str], segments: Sequence[str]) -> bool:
    """Match ``**``-aware *patterns* against path *segments*."""
    if not patterns:
        return not segments
    head, rest = patterns[0], patterns[1:]
    if head == "**":
        if not rest:
            return True
        return any(_glob_segments(rest, segments[i:]) for i in range(len(segments) + 1))
    if not segments:
        return False
    if not fnmatch.fnmatchcase(segments[0], head):
        return False
    return _glob_segments(rest, segments[1:])


# --------------------------------------------------------------------------
# access: is this resolved resource inside the grant?
# --------------------------------------------------------------------------


def resolve_fs_path(raw: str | Path, workspace: str | Path) -> Path:
    """Resolve *raw* exactly the way the capability will open it.

    Relative paths are taken against *workspace*; ``..`` is collapsed and
    symlinks are followed, so a link pointing out of a granted directory
    resolves to its real target and can be denied. ``~`` is deliberately NOT
    expanded, because :func:`open` does not expand it either -- the check must
    describe the byte path that will actually be opened.
    """
    path = Path(raw)
    if not path.is_absolute():
        path = Path(workspace) / path
    return path.resolve()


def _grant_root(prefix: str, workspace: str | Path) -> Path:
    root = Path(prefix) if prefix else Path(workspace)
    if not root.is_absolute():
        root = Path(workspace) / root
    return root.resolve()


def _one_grant_covers(pattern: str, resolved: Path, workspace: str | Path) -> bool:
    prefix, tail = _split_pattern(pattern)
    root = _grant_root(prefix, workspace)
    if not tail:
        return resolved == root or resolved.is_relative_to(root)
    if not resolved.is_relative_to(root):
        return False
    if resolved == root:
        rel_segments: list[str] = []
    else:
        rel_segments = list(resolved.relative_to(root).parts)
    return _glob_segments(tail, rel_segments)


def path_within_grants(
    patterns: Iterable[str], resolved: Path, workspace: str | Path
) -> bool:
    """True when *resolved* is covered by at least one grant in *patterns*.

    *resolved* must already have been through :func:`resolve_fs_path`.
    """
    for pattern in patterns:
        if not pattern:
            continue
        if _one_grant_covers(pattern, resolved, workspace):
            return True
    return False


def assert_path_within_grants(
    patterns: Iterable[str], resolved: Path, workspace: str | Path, what: str
) -> None:
    """Fail-closed variant of :func:`path_within_grants`."""
    if not path_within_grants(patterns, resolved, workspace):
        raise AuthorityError(f"{what}: {resolved} is outside granted authority {tuple(patterns)!r}")


# --------------------------------------------------------------------------
# delegation: may a child hold this pattern at all?
# --------------------------------------------------------------------------


def pattern_covers(grant: str, candidate: str) -> bool:
    """Whole-token match used for non-path dimensions."""
    if not grant or not candidate:
        return False
    if grant == candidate:
        return True
    return fnmatch.fnmatchcase(candidate, grant)


def _is_filesystem_wide(pattern: str) -> bool:
    """``/**`` -- the only way to ask for the whole filesystem."""
    prefix, tail = _split_pattern(pattern)
    return prefix == "/" and "**" in tail


def _pattern_scope_covers(grant: str, candidate: str) -> bool:
    """True when filesystem pattern *candidate* is no wider than *grant*."""
    if not grant or not candidate:
        return False
    if grant == candidate:
        return True
    # A filesystem-wide grant subsumes every other pattern, including the
    # workspace-relative "**".
    if _is_filesystem_wide(grant):
        return True
    if _is_filesystem_wide(candidate):
        return False

    grant_prefix, grant_tail = _split_pattern(grant)
    cand_prefix, cand_tail = _split_pattern(candidate)

    grant_root = PurePosixPath(grant_prefix)
    cand_root = PurePosixPath(cand_prefix)

    # Abstract patterns are never resolved against a filesystem, so traversal
    # cannot be collapsed safely here. Refuse it outright.
    if ".." in grant_root.parts or ".." in cand_root.parts:
        return False
    if grant_root.is_absolute() != cand_root.is_absolute():
        return False

    grant_parts = grant_root.parts if grant_prefix not in ("", ".") else ()
    cand_parts = cand_root.parts if cand_prefix not in ("", ".") else ()
    if cand_parts[: len(grant_parts)] != grant_parts:
        return False

    if not grant_tail:
        # A literal prefix grant covers its whole subtree.
        return True
    if "**" in grant_tail:
        return True
    # The grant is depth-bounded, so the candidate must not reach deeper.
    if "**" in cand_tail:
        return False
    depth_under_grant = len(cand_parts) - len(grant_parts)
    return depth_under_grant + len(cand_tail) <= len(grant_tail)


def authority_covers(parent: "Authority", child: "Authority") -> bool:
    """True when every power *child* requests is already held by *parent*.

    This is the delegation rule from #492: "children can never widen authority."
    """
    for dimension in DIMENSIONS:
        granted = tuple(getattr(parent, dimension))
        requested = tuple(getattr(child, dimension))
        for candidate in requested:
            if not candidate:
                return False
            if dimension in FS_DIMENSIONS:
                covered = any(_pattern_scope_covers(g, candidate) for g in granted)
            else:
                covered = any(pattern_covers(g, candidate) for g in granted)
            if not covered:
                return False
    return True


def missing_powers(parent: "Authority", child: "Authority") -> list[str]:
    """Every ``dimension:pattern`` in *child* that *parent* does not cover."""
    missing: list[str] = []
    for dimension in DIMENSIONS:
        granted = tuple(getattr(parent, dimension))
        for candidate in getattr(child, dimension):
            if dimension in FS_DIMENSIONS:
                covered = bool(candidate) and any(
                    _pattern_scope_covers(g, candidate) for g in granted
                )
            else:
                covered = bool(candidate) and any(pattern_covers(g, candidate) for g in granted)
            if not covered:
                missing.append(f"{dimension}:{candidate}")
    return missing

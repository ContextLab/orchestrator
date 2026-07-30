"""Model-name resolution for the Anthropic adapter.

`_normalize_model_name` used to substring-match the family name and rewrite
*every* id containing "haiku"/"opus"/"sonnet" to a hard-coded 2024 model, so

    AnthropicModel(name="claude-haiku-4-5-20251001")

actually requested `claude-3-haiku-20240307` and the API returned
404 not_found_error. No current Claude model was reachable, and the caller's
explicit choice was discarded silently.

These tests are hermetic: they exercise resolution only, never the network.
Whether the alias *targets* exist is a question only the live API can answer,
and that is asserted in tests/test_live_anthropic.py.
"""

import pytest

from orchestrator.models.anthropic_model import AnthropicModel

pytestmark = pytest.mark.unit


def _resolve(name: str) -> str:
    """Call the resolver without constructing a client (no API key needed)."""
    return AnthropicModel._normalize_model_name(AnthropicModel, name)


@pytest.mark.parametrize(
    "name",
    [
        # Dated ids across generations -- all must survive untouched.
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-5-20250929",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        # Rolling aliases are already fully qualified.
        "claude-3-5-sonnet-latest",
        "claude-opus-4-latest",
    ],
)
def test_qualified_ids_pass_through_unchanged(name):
    """The regression that broke every current model.

    A caller who names an exact model must get that exact model. Silently
    substituting a different one is wrong even when the substitute works.
    """
    assert _resolve(name) == name


@pytest.mark.parametrize("family", ["haiku", "opus", "sonnet"])
def test_bare_family_names_resolve_to_an_alias(family):
    resolved = _resolve(family)
    assert resolved == AnthropicModel._FAMILY_ALIASES[family]
    assert resolved != family, "a bare family name is not a servable model id"


def test_family_aliases_are_rolling_not_pinned():
    """Pinned dates are what made the previous table rot.

    A dated alias is correct when written and 404s once retired. `-latest`
    cannot go stale, so the table does not need maintenance to keep working.
    """
    for family, alias in AnthropicModel._FAMILY_ALIASES.items():
        assert alias.endswith("-latest"), (
            f"{family!r} maps to the pinned id {alias!r}; use a -latest alias "
            "so it cannot expire"
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [("claude-2.1", "claude-2.1"), ("claude-2", "claude-2.0"),
     ("claude-instant", "claude-instant-1.2")],
)
def test_legacy_generations_keep_their_exact_ids(name, expected):
    assert _resolve(name) == expected


def test_unknown_names_are_passed_through_not_guessed():
    """An unknown id must reach the API and produce a clear 404.

    Substituting a "closest match" turns a one-line error into a silent
    behaviour change that is very hard to notice.
    """
    assert _resolve("totally-unknown-model") == "totally-unknown-model"


def test_family_match_does_not_hijack_a_qualified_id():
    """The exact shape of the original bug, pinned so it cannot return."""
    name = "claude-haiku-4-5-20251001"
    assert "haiku" in name, "test premise: the id contains a family name"
    assert _resolve(name) == name, (
        "a qualified id containing a family name was rewritten -- this is the "
        "regression that made every current Claude model unreachable"
    )

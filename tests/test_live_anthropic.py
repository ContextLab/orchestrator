"""Live acceptance test for the Anthropic provider contract.

This is the only test that spends money. It exists because everything else in
the suite is hermetic: without it, "Anthropic is the supported provider" is an
untested claim. Before this file existed, `pytest -m live` collected zero
tests and exited successfully, so the live CI job proved nothing.

Scope is deliberately minimal -- one cheap call on the cheapest model, with a
hard output cap -- because the point is to verify the *contract* (we call the
API correctly and get back the shape we expect), not model quality.

Run with:
    ANTHROPIC_API_KEY=... pytest -m live -v

Without a key, `tests/conftest.py` skips these with a reason naming the
missing variable. They never silently pass.
"""

import os

import pytest

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

# The cheapest current model. Kept as a module constant so a failure report
# names exactly what was exercised.
LIVE_MODEL = "claude-haiku-4-5-20251001"

# A hard ceiling on spend per call. The prompts below want a handful of
# tokens; this is loose enough not to truncate a correct answer and tight
# enough that a runaway loop cannot become expensive.
MAX_TOKENS = 32


def _require_anthropic_package():
    """Import the anthropic SDK, or skip -- unless live coverage is required.

    In the live CI job a missing extra must FAIL, not skip. Otherwise a broken
    install makes the job pass green having executed nothing, which is exactly
    the hole this file was written to close.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        message = (
            f"the [anthropic] extra is not installed ({exc}); "
            "install with: pip install 'py-orc[anthropic]'"
        )
        if os.environ.get("ORCHESTRATOR_REQUIRE_LIVE") == "1":
            pytest.fail(
                "ORCHESTRATOR_REQUIRE_LIVE=1 demands real live coverage but "
                + message
            )
        pytest.skip(message)


#: Substrings identifying an account/billing precondition rather than a defect
#: in this codebase. These must not be reported as our failure: doing so sends
#: someone hunting a bug in the adapter when the real fix is to add credit.
_ACCOUNT_PRECONDITIONS = (
    "credit balance is too low",
    "quota",
    "rate_limit",
)


def _skip_if_account_blocked(exc: Exception) -> None:
    """Skip (or fail, under REQUIRE_LIVE) on a billing/quota precondition."""
    message = str(exc)
    if not any(marker in message.lower() for marker in _ACCOUNT_PRECONDITIONS):
        return
    reason = (
        "the Anthropic account cannot serve requests (billing/quota), so live "
        f"provider behaviour was NOT verified: {message}"
    )
    if os.environ.get("ORCHESTRATOR_REQUIRE_LIVE") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _model():
    """Build a model against the real API, skipping with a precise reason."""
    _require_anthropic_package()
    from orchestrator.models.anthropic_model import AnthropicModel

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert api_key, "conftest should have skipped: ANTHROPIC_API_KEY is unset"

    # use_langchain=False exercises the direct Anthropic path, so this test
    # depends only on the [anthropic] extra rather than the langgraph stack.
    return AnthropicModel(name=LIVE_MODEL, api_key=api_key, use_langchain=False)


async def test_generate_returns_text_from_the_real_api():
    """The provider contract: generate(prompt) -> non-empty str."""
    model = _model()

    try:
        response = await model.generate(
            prompt="Reply with exactly the word: pong",
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:
        _skip_if_account_blocked(exc)
        raise

    # Report what was actually exercised, so a CI failure is diagnosable
    # without re-running against a paid API.
    print(f"\nlive model: {LIVE_MODEL}\nlive response: {response!r}")

    assert isinstance(response, str), (
        f"generate() must return str, got {type(response).__name__}"
    )
    assert response.strip(), "generate() returned an empty response"
    # Deliberately loose: this asserts the round trip worked, not that the
    # model is obedient. A strict equality check here would make the test
    # flaky for reasons that say nothing about our code.
    assert "pong" in response.lower(), (
        f"expected the model to echo 'pong', got {response!r}"
    )


async def test_generate_respects_max_tokens():
    """max_tokens must actually reach the API, not be silently dropped.

    A provider adapter that ignores max_tokens is both a correctness bug and a
    cost bug, and it is invisible to every hermetic test.
    """
    model = _model()

    try:
        response = await model.generate(
            prompt="Count slowly from 1 to 500, one number per line.",
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:
        _skip_if_account_blocked(exc)
        raise

    print(f"\nlive model: {LIVE_MODEL}\ntruncated length: {len(response)} chars")

    assert isinstance(response, str)
    # 32 tokens cannot render 500 numbers. Generous char bound: even at ~6
    # chars/token this is far under what an uncapped answer would produce.
    assert len(response) < 600, (
        f"max_tokens={MAX_TOKENS} appears not to have been applied; "
        f"got {len(response)} chars"
    )


async def test_every_family_alias_resolves_to_a_real_model():
    """Each bare family name must resolve to a model the API actually serves.

    Resolution goes through the Models API, so only a live run can prove it
    works -- a hermetic test could only assert that the code calls the code.
    This is the test that caught two successive rounds of hard-coded ids being
    wrong: pinned 2024 ids (retired) and invented "-latest" aliases (never
    existed), both 404.
    """
    _require_anthropic_package()
    from orchestrator.models.anthropic_model import AnthropicModel

    failures = []
    for family in AnthropicModel._FAMILIES:
        model = AnthropicModel(
            name=family, api_key=os.environ["ANTHROPIC_API_KEY"], use_langchain=False
        )
        try:
            reply = await model.generate(
                prompt="Reply with the single word: ok",
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )
            resolved = AnthropicModel._family_cache.get(family, "<unresolved>")
            print(f"\nfamily {family!r} -> {resolved!r}: OK ({reply.strip()[:40]!r})")
        except Exception as exc:  # noqa: BLE001 - reporting all failures at once
            _skip_if_account_blocked(exc)
            failures.append(f"{family!r}: {exc}")

    assert not failures, (
        "these model families could not be resolved to a servable model:\n  "
        + "\n  ".join(failures)
    )


async def test_models_api_lists_servable_models():
    """Record what the account can actually serve.

    Printed so a CI log is a primary source for which ids exist, instead of
    the guesswork that produced two rounds of 404s in this adapter.
    """
    _require_anthropic_package()
    from orchestrator.models.providers.anthropic_provider import AnthropicProvider
    from orchestrator.models.providers.base import ProviderConfig

    provider = AnthropicProvider(
        ProviderConfig(name="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
    )
    await provider.initialize()
    models = await provider.discover_models()

    print("\nmodels served to this account:")
    for model_id in sorted(models):
        print(f"  {model_id}")

    assert models, "the Models API returned no models"
    assert any("claude" in m.lower() for m in models)


async def test_health_check_reports_true_against_the_real_api():
    """The provider's own readiness probe must agree with reality."""
    _require_anthropic_package()
    from orchestrator.models.providers.anthropic_provider import AnthropicProvider
    from orchestrator.models.providers.base import ProviderConfig

    provider = AnthropicProvider(
        ProviderConfig(name="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
    )
    await provider.initialize()

    healthy = await provider.health_check()
    print(f"\nlive provider health_check: {healthy}")

    assert healthy is True, (
        "health_check() returned falsey against a real API with a valid key"
    )

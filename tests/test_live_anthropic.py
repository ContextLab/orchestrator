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

    response = await model.generate(
        prompt="Reply with exactly the word: pong",
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )

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

    response = await model.generate(
        prompt="Count slowly from 1 to 500, one number per line.",
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )

    print(f"\nlive model: {LIVE_MODEL}\ntruncated length: {len(response)} chars")

    assert isinstance(response, str)
    # 32 tokens cannot render 500 numbers. Generous char bound: even at ~6
    # chars/token this is far under what an uncapped answer would produce.
    assert len(response) < 600, (
        f"max_tokens={MAX_TOKENS} appears not to have been applied; "
        f"got {len(response)} chars"
    )


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

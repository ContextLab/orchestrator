"""Live acceptance tests for the Dartmouth Chat provider.

Unlike the Anthropic live tests, these cost **nothing**: they only ever use
models the live catalog reports at zero cost per token, and they assert that
before making a request. They are still marked ``live`` because they need a
credential and the network.

Run with:
    pytest -m live -k dartmouth -v
"""

import pytest

from orchestrator.models.dartmouth_credentials import resolve_dartmouth_api_key
from orchestrator.models.dartmouth_model import (
    ALLOW_PAID_ENV_VAR,
    ModelUnavailable,
    PaidModelRefused,
    ReasoningTruncated,
)
from orchestrator.models.providers.dartmouth_provider import DartmouthProvider

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

#: Small, fast, and free. Used where the test is about the round trip rather
#: than the model.
SMALL_FREE_MODEL = "meta.llama-3-2-3b-instruct"


def _require_credential():
    credential = resolve_dartmouth_api_key(required=False)
    if credential is None:
        pytest.skip(
            "no Dartmouth Chat credential; set DARTMOUTH_CHAT_API_KEY or add "
            "one to ~/.orchestrator/.env"
        )
    return credential


async def _provider() -> DartmouthProvider:
    _require_credential()
    provider = DartmouthProvider()
    await provider.initialize()
    return provider


async def test_catalog_reports_free_models():
    """The whole premise: this gateway serves models that cost nothing."""
    provider = await _provider()
    free = provider.list_free_models()

    print(f"\nfree models ({len(free)}):")
    for model_id in free:
        print(f"  {model_id}")
    print(f"paid models: {len(provider.list_paid_models())}")

    assert free, (
        "the catalog reported no free models -- if Dartmouth changed its "
        "pricing this project's free-first assumption no longer holds"
    )
    assert SMALL_FREE_MODEL in free, (
        f"{SMALL_FREE_MODEL} is no longer free; pick another from {free}"
    )


async def test_generate_against_a_free_model():
    provider = await _provider()
    model = await provider.create_model(SMALL_FREE_MODEL)

    assert model.cost.is_free, "refusing to spend money in a test"
    assert await model.estimate_cost("hello") == 0.0

    reply = await model.generate(
        "Reply with exactly the word: pong", temperature=0.0, max_tokens=32
    )
    print(f"\n{SMALL_FREE_MODEL} -> {reply.strip()[:80]!r}")

    assert isinstance(reply, str) and reply.strip()
    assert "pong" in reply.lower()


async def test_reasoning_model_answers_when_given_headroom():
    """The default token budget must be enough for a reasoning model.

    qwen3.5-122b emits `reasoning_content` before `content`. At max_tokens=16
    it burned 307 completion tokens thinking and returned content=null. This
    asserts the default budget avoids that.
    """
    provider = await _provider()
    free = provider.list_free_models()
    reasoning = next((m for m in free if "qwen3.5" in m), None)
    if reasoning is None:
        pytest.skip(f"no qwen3.5 reasoning model in the free set: {free}")

    model = await provider.create_model(reasoning)
    try:
        reply = await model.generate(
            "Reply with exactly the word: pong", temperature=0.0
        )
    except ModelUnavailable as exc:
        # Free models are served from individual cluster endpoints that go
        # down independently. That is an upstream outage, not a defect here.
        pytest.skip(f"{reasoning} backend is down: {exc}")
    print(f"\n{reasoning} -> {reply.strip()[:80]!r}")

    assert "pong" in reply.lower()


async def test_truncated_reasoning_reports_the_real_cause():
    """A starved reasoning model must raise, not return an empty string."""
    provider = await _provider()
    reasoning = next(
        (m for m in provider.list_free_models() if "qwen3.5" in m), None
    )
    if reasoning is None:
        pytest.skip("no qwen3.5 reasoning model available")

    model = await provider.create_model(reasoning)
    # ModelUnavailable subclasses DartmouthModelError, so it must be caught
    # BEFORE asserting on the truncation error -- otherwise a backend outage
    # is silently mistaken for the condition under test.
    try:
        await model.generate("Explain quantum computing.", max_tokens=8)
    except ModelUnavailable as exc:
        pytest.skip(f"{reasoning} backend is down: {exc}")
    except ReasoningTruncated as exc:
        assert "max_tokens" in str(exc), "the error must name the fix"
        return
    pytest.fail("a starved reasoning model returned instead of raising")


async def test_structured_output_parses_as_json():
    provider = await _provider()
    model = await provider.create_model(SMALL_FREE_MODEL)

    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
        },
        "required": ["city", "country"],
    }
    result = await model.generate_structured(
        "The capital of France.", schema=schema, temperature=0.0
    )
    print(f"\nstructured -> {result!r}")

    assert isinstance(result, dict)
    assert "paris" in str(result).lower()


async def test_paid_models_are_refused_without_optin(monkeypatch):
    """The guard must hold against the real catalog, not just fixtures."""
    monkeypatch.delenv(ALLOW_PAID_ENV_VAR, raising=False)
    provider = await _provider()

    paid = provider.list_paid_models()
    if not paid:
        pytest.skip("catalog lists no paid models to test the guard against")

    with pytest.raises(PaidModelRefused):
        await provider.create_model(paid[0])


async def test_provider_health_check():
    provider = await _provider()
    assert await provider.health_check() is True


async def test_generate_free_falls_back_past_downed_models():
    """The reason this helper exists.

    Individual free-model endpoints flap. A caller that names one model is
    stranded whenever that model is down; `generate_free` walks the
    preference order until something answers.
    """
    provider = await _provider()

    reply, model_id = await provider.generate_free(
        "Reply with exactly the word: pong", temperature=0.0, max_tokens=64
    )
    print(f"\ngenerate_free answered via {model_id} -> {reply.strip()[:60]!r}")

    assert "pong" in reply.lower()
    assert model_id in provider.list_free_models(), "fallback must stay free"


async def test_free_preference_order_is_complete_and_free():
    """Every free model appears exactly once, and nothing paid sneaks in."""
    provider = await _provider()
    ordered = provider.free_models_by_preference()

    assert sorted(ordered) == sorted(provider.list_free_models())
    assert len(ordered) == len(set(ordered)), "a model appears twice"
    paid = set(provider.list_paid_models())
    assert not (set(ordered) & paid), "a paid model leaked into the free chain"

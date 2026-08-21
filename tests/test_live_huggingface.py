"""Live acceptance tests for the HuggingFace Inference API provider.

These need a token and the network. Generation tests only ever use models the
live catalog reports with a free route (an ``is_free`` promo or explicit zero
pricing), pinned to that provider so the router cannot bill the account.

Two lessons carried over from the Dartmouth live suite:

1. **Flapping free endpoints skip, not fail.** A free route is a promo on one
   provider and it flaps; an upstream outage is not a defect in this adapter.
   The CI job compensates by requiring at least one real pass.
2. **A reasoning model that spends its whole budget thinking is not a pass.**
   Empty ``content`` with only ``reasoning_content`` must surface as
   ``ReasoningTruncated`` -- observed live on this router, where
   prism-ml/Ternary-Bonsai-27B-AWQ-4bit at max_tokens=32 returned 31
   reasoning tokens and no answer.

An invalid token is NOT a flap: it fails, so a misconfigured CI secret turns
the job red instead of green-with-skips.

Run with:
    pytest -m live -k huggingface -v
"""

import pytest

from orchestrator.models.huggingface_credentials import resolve_huggingface_api_key
from orchestrator.models.huggingface_model import (
    ALLOW_PAID_ENV_VAR,
    HuggingFaceModelError,
    ModelLoading,
    ModelUnavailable,
    PaidModelRefused,
    PaymentRequired,
    RateLimited,
    ReasoningTruncated,
)
from orchestrator.models.providers.huggingface_provider import HuggingFaceProvider

pytestmark = [pytest.mark.live, pytest.mark.asyncio]


def _require_credential():
    credential = resolve_huggingface_api_key(required=False)
    if credential is None:
        pytest.skip(
            "no HuggingFace token; set HF_TOKEN, add one to "
            "~/.orchestrator/.env, or run `hf auth login`"
        )
    return credential


async def _provider() -> HuggingFaceProvider:
    _require_credential()
    provider = HuggingFaceProvider()
    await provider.initialize()
    return provider


def _skip_if_flapping(exc: Exception, model_id: str):
    """Upstream conditions that must skip rather than fail. 401 is not one."""
    if isinstance(exc, RateLimited):
        pytest.skip(f"account is rate-limited: {exc}")
    if isinstance(exc, PaymentRequired):
        pytest.skip(f"account credits are depleted (monthly reset): {exc}")
    if isinstance(exc, (ModelLoading, ModelUnavailable)):
        pytest.skip(f"{model_id} is not serving right now: {exc}")


async def test_catalog_reports_models_and_free_routes():
    """The premise: the router serves chat models, some with a free route."""
    provider = await _provider()
    models = await provider.discover_models()
    free = provider.list_free_models()

    print(f"\ncatalog: {len(models)} chat models, {len(free)} with a free route:")
    for model_id in free:
        print(f"  {model_id}")
    print(f"paid/unknown: {len(provider.list_paid_models())}")

    assert models, "the router catalog returned no chat models at all"


async def test_generate_against_a_free_routed_model():
    """Proves the token can actually run inference, at zero cost."""
    provider = await _provider()
    free = provider.list_free_models()
    if not free:
        pytest.skip("no model currently has a free route (promo drought)")

    model_id = provider.free_models_by_preference()[0]
    model = await provider.create_model(model_id)
    try:
        assert model.cost.is_free, "refusing to spend money in a test"
        assert await model.estimate_cost("hello") == 0.0
        reply = await model.generate(
            "Reply with exactly the word: pong", temperature=0.0
        )
    except (RateLimited, PaymentRequired, ModelUnavailable, ReasoningTruncated) as exc:
        _skip_if_flapping(exc, model_id)
        raise
    finally:
        await model.aclose()
    print(f"\n{model_id} -> {reply.strip()[:80]!r}")

    assert isinstance(reply, str) and reply.strip(), (
        "an empty reply is not a pass -- see the reasoning-truncation lesson"
    )
    assert "pong" in reply.lower()


async def test_truncated_reasoning_reports_the_real_cause():
    """A starved reasoning model must raise, not return an empty string."""
    provider = await _provider()
    free = provider.list_free_models()
    if not free:
        pytest.skip("no model currently has a free route")

    # A non-reasoning model starved at max_tokens=8 still emits partial
    # content, so only a reasoning model can demonstrate this. Try a few.
    tried = []
    for model_id in provider.free_models_by_preference()[:3]:
        model = await provider.create_model(model_id)
        try:
            # ModelUnavailable subclasses HuggingFaceModelError, so transient
            # outages are caught and skipped BEFORE the truncation assertion.
            await model.generate("Explain quantum computing.", max_tokens=8)
        except (ModelUnavailable, RateLimited, PaymentRequired) as exc:
            _skip_if_flapping(exc, model_id)
        except ReasoningTruncated as exc:
            assert "max_tokens" in str(exc), "the error must name the fix"
            return
        finally:
            await model.aclose()
        tried.append(model_id)
    pytest.skip(f"no reasoning model among the current free routes: {tried}")


async def test_paid_models_are_refused_without_optin(monkeypatch):
    """The guard must hold against the real catalog, not just fixtures."""
    monkeypatch.delenv(ALLOW_PAID_ENV_VAR, raising=False)
    provider = await _provider()

    paid = provider.list_paid_models()
    assert paid, "a 130-model catalog with nothing paid would be suspicious"

    with pytest.raises(PaidModelRefused):
        await provider.create_model(paid[0])


async def test_paid_pricing_flows_from_the_real_catalog(monkeypatch):
    """Per-million USD in the catalog must arrive as per-1k on the model."""
    monkeypatch.setenv(ALLOW_PAID_ENV_VAR, "1")
    provider = await _provider()
    # Only a model with explicit catalog pricing can prove the flow; an
    # unpriced entry is unknown-priced and its estimate rightly refuses.
    priced = [
        m
        for m in provider.list_paid_models()
        if provider.get_model_cost(m).output_cost_per_1k_tokens > 0
    ]
    if not priced:
        pytest.skip("catalog currently reports no explicit prices")

    model = await provider.create_model(priced[0])
    estimate = await model.estimate_cost("hello", max_tokens=1000)
    print(f"\n{priced[0]}: 1000-token estimate ${estimate:.6f}")
    assert estimate > 0.0


async def test_provider_health_check():
    provider = await _provider()
    assert await provider.health_check() is True


async def test_generate_free_answers_via_the_fallback_chain():
    """The reason this helper exists: free routes flap, demos must not."""
    provider = await _provider()
    if not provider.list_free_models():
        pytest.skip("no model currently has a free route (promo drought)")

    try:
        reply, model_id = await provider.generate_free(
            "Reply with exactly the word: pong", temperature=0.0
        )
    except RateLimited as exc:
        pytest.skip(f"account is rate-limited: {exc}")
    except PaymentRequired as exc:
        pytest.skip(f"account credits are depleted (monthly reset): {exc}")
    print(f"\ngenerate_free answered via {model_id} -> {reply.strip()[:60]!r}")

    assert reply.strip(), "an empty reply is not a pass"
    assert "pong" in reply.lower()
    assert model_id in provider.list_free_models(), "fallback must stay free"


async def test_free_preference_covers_the_whole_free_set():
    """Every free model appears exactly once, and nothing paid sneaks in."""
    provider = await _provider()
    ordered = provider.free_models_by_preference()

    assert sorted(ordered) == sorted(provider.list_free_models())
    assert len(ordered) == len(set(ordered)), "a model appears twice"
    paid = set(provider.list_paid_models())
    assert not (set(ordered) & paid), "a paid model leaked into the free chain"

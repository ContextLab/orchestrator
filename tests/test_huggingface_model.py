"""Unit and contract tests for the HuggingFace Inference API adapter.

Hermetic throughout: the router is faked with deterministic stand-ins (a
recording ``_post``, a fake aiohttp session, a static catalog), per ADR 0001's
test-layer policy. Live acceptance lives in ``test_live_huggingface.py``.

The wire contract under test (verified against the live router, 2026-08-21):

- ``POST https://router.huggingface.co/v1/chat/completions`` with a
  ``Bearer $HF_TOKEN`` header; OpenAI-compatible request/response.
- ``GET /v1/models`` lists chat models; each entry carries ``providers`` with
  ``status`` (``live``/``error``), ``pricing`` in USD **per million** tokens,
  an ``is_free`` promo flag, ``context_length`` and ``throughput``.
- A reasoning model that exhausts its budget returns ``content`` absent and
  ``reasoning_content`` present with ``finish_reason: "length"`` -- observed
  live with ``prism-ml/Ternary-Bonsai-27B-AWQ-4bit`` at ``max_tokens=32``
  (31 of 32 completion tokens were reasoning). That is truncation, not an
  empty answer.
"""

import asyncio
import json

import pytest

from orchestrator.core.model import ModelCapabilities, ModelCost
from orchestrator.models.huggingface_model import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    HuggingFaceInferenceModel,
    HuggingFaceModelError,
    InsecureEndpoint,
    ModelLoading,
    ModelUnavailable,
    PaidModelRefused,
    PaymentRequired,
    RateLimited,
    ReasoningTruncated,
    ReservedRequestField,
    validate_base_url,
)

pytestmark = pytest.mark.unit

FAKE_KEY = "hf_0123456789abcdef0123456789abcdef"

FREE = ModelCost(is_free=True)
PAID = ModelCost(input_cost_per_1k_tokens=0.001, output_cost_per_1k_tokens=0.002)


@pytest.fixture(autouse=True)
def _no_paid_optin(monkeypatch):
    """The paid opt-in must never leak in from the developer's shell."""
    monkeypatch.delenv("ORCHESTRATOR_ALLOW_PAID_MODELS", raising=False)


def _free_model(**kwargs):
    return HuggingFaceInferenceModel(
        name="prism-ml/Ternary-Bonsai-27B-gguf",
        api_key=FAKE_KEY,
        cost=FREE,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Cost policy -- unknown pricing is treated as paid, never as free
# ---------------------------------------------------------------------------

def test_free_model_constructs_without_optin():
    assert _free_model().cost.is_free


def test_paid_model_is_refused_by_default():
    with pytest.raises(PaidModelRefused) as excinfo:
        HuggingFaceInferenceModel(name="x/y", api_key=FAKE_KEY, cost=PAID)
    assert "ORCHESTRATOR_ALLOW_PAID_MODELS" in str(excinfo.value)


def test_paid_model_allowed_with_explicit_optin(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_PAID_MODELS", "1")
    model = HuggingFaceInferenceModel(name="x/y", api_key=FAKE_KEY, cost=PAID)
    assert not model.cost.is_free


@pytest.mark.parametrize("value", ["0", "true", "yes", "2", ""])
def test_only_the_exact_value_1_enables_paid_usage(monkeypatch, value):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_PAID_MODELS", value)
    with pytest.raises(PaidModelRefused):
        HuggingFaceInferenceModel(name="x/y", api_key=FAKE_KEY, cost=PAID)


def test_unpriced_model_is_refused_rather_than_assumed_free():
    """A model built without catalog pricing has an *unknown* cost."""
    with pytest.raises(PaidModelRefused):
        HuggingFaceInferenceModel(name="x/y", api_key=FAKE_KEY)


def test_unpriced_model_refuses_to_estimate_cost(monkeypatch):
    """A zero-filled default would report $0.00 for a model that may bill."""
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_PAID_MODELS", "1")
    model = HuggingFaceInferenceModel(name="x/y", api_key=FAKE_KEY)
    with pytest.raises(HuggingFaceModelError, match="without pricing"):
        asyncio.run(model.estimate_cost("hello"))


def test_estimate_cost_is_exactly_zero_for_free_models():
    model = _free_model()
    assert asyncio.run(model.estimate_cost("hello")) == 0.0


def test_estimate_cost_uses_real_pricing_for_paid_models(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_PAID_MODELS", "1")
    model = HuggingFaceInferenceModel(name="x/y", api_key=FAKE_KEY, cost=PAID)
    estimate = asyncio.run(model.estimate_cost("hello", max_tokens=1000))
    assert estimate > 0.0


# ---------------------------------------------------------------------------
# Catalog pricing -- the router reports USD per *million* tokens, per provider
# ---------------------------------------------------------------------------

def _entry(model_id, providers, modalities=("text",)):
    return {
        "id": model_id,
        "object": "model",
        "created": 1,
        "owned_by": model_id.split("/")[0],
        "architecture": {
            "input_modalities": list(modalities),
            "output_modalities": ["text"],
        },
        "providers": providers,
    }


def _provider(name, *, status="live", pricing=..., is_free=..., throughput=None):
    entry = {"provider": name, "status": status}
    if pricing is not ...:
        entry["pricing"] = pricing
    if is_free is not ...:
        entry["is_free"] = is_free
    if throughput is not None:
        entry["throughput"] = throughput
    return entry


def test_zero_priced_live_provider_makes_the_model_free():
    from orchestrator.models.providers.huggingface_provider import (
        model_cost_from_catalog,
    )

    entry = _entry("a/b", [_provider("together", pricing={"input": 0, "output": 0})])
    assert model_cost_from_catalog(entry).is_free


def test_is_free_promo_flag_makes_the_model_free_without_pricing():
    from orchestrator.models.providers.huggingface_provider import (
        model_cost_from_catalog,
    )

    entry = _entry("a/b", [_provider("groq", is_free=True)])
    assert model_cost_from_catalog(entry).is_free


def test_priced_model_is_not_free_and_converts_per_million_to_per_1k():
    from orchestrator.models.providers.huggingface_provider import (
        model_cost_from_catalog,
    )

    entry = _entry(
        "a/b", [_provider("novita", pricing={"input": 1.69, "output": 3.38})]
    )
    cost = model_cost_from_catalog(entry)
    assert not cost.is_free
    assert cost.input_cost_per_1k_tokens == pytest.approx(0.00169)
    assert cost.output_cost_per_1k_tokens == pytest.approx(0.00338)


def test_paid_cost_uses_the_most_expensive_live_route():
    """Routing is server-side; budgeting on the cheapest route understates."""
    from orchestrator.models.providers.huggingface_provider import (
        model_cost_from_catalog,
    )

    entry = _entry(
        "a/b",
        [
            _provider("cheap", pricing={"input": 1.0, "output": 2.0}),
            _provider("dear", pricing={"input": 5.0, "output": 4.0}),
        ],
    )
    cost = model_cost_from_catalog(entry)
    assert cost.input_cost_per_1k_tokens == pytest.approx(0.005)
    assert cost.output_cost_per_1k_tokens == pytest.approx(0.004)


def test_unpriced_entry_is_treated_as_paid():
    """Absence of a price is not evidence of zero price."""
    from orchestrator.models.providers.huggingface_provider import (
        model_cost_from_catalog,
    )

    assert not model_cost_from_catalog(_entry("a/b", [_provider("novita")])).is_free
    assert not model_cost_from_catalog(_entry("a/b", [])).is_free


def test_a_single_nonzero_price_makes_a_model_paid():
    from orchestrator.models.providers.huggingface_provider import (
        model_cost_from_catalog,
    )

    entry = _entry("a/b", [_provider("x", pricing={"input": 0, "output": 0.5})])
    assert not model_cost_from_catalog(entry).is_free


def test_providers_in_error_state_do_not_count():
    """A free route that is down must not mark the model free."""
    from orchestrator.models.providers.huggingface_provider import (
        model_cost_from_catalog,
    )

    entry = _entry(
        "a/b",
        [
            _provider("together", status="error", pricing={"input": 0, "output": 0}),
            _provider("novita", pricing={"input": 1.0, "output": 1.0}),
        ],
    )
    assert not model_cost_from_catalog(entry).is_free


def test_free_route_is_pinned_to_the_free_provider():
    """An unpinned request routes :fastest -- which may be a paid provider."""
    from orchestrator.models.providers.huggingface_provider import (
        free_route_from_catalog,
    )

    entry = _entry(
        "a/b",
        [
            _provider("novita", pricing={"input": 1.0, "output": 1.0}),
            _provider("together", pricing={"input": 0, "output": 0}),
        ],
    )
    assert free_route_from_catalog(entry) == "together"


def test_paid_model_has_no_free_route():
    from orchestrator.models.providers.huggingface_provider import (
        free_route_from_catalog,
    )

    entry = _entry("a/b", [_provider("novita", pricing={"input": 1, "output": 1})])
    assert free_route_from_catalog(entry) is None


def test_only_free_models_are_selected_for_registration():
    """A paid model registered as if free is the accident that costs money."""
    from orchestrator.models.providers.huggingface_provider import (
        free_models_from_catalog,
    )

    catalog = {
        "free/zero": _entry(
            "free/zero", [_provider("together", pricing={"input": 0, "output": 0})]
        ),
        "free/promo": _entry("free/promo", [_provider("groq", is_free=True)]),
        "paid/model": _entry(
            "paid/model", [_provider("novita", pricing={"input": 1, "output": 2})]
        ),
        "unknown/model": _entry("unknown/model", [_provider("novita")]),
    }

    free = free_models_from_catalog(catalog)

    assert set(free) == {"free/zero", "free/promo"}
    assert all(cost.is_free for cost in free.values())


# ---------------------------------------------------------------------------
# Response extraction -- a reasoning model's scratchpad is not an answer
# ---------------------------------------------------------------------------

def test_plain_content_is_returned():
    response = {"choices": [{"message": {"content": "pong"}}]}
    assert (
        HuggingFaceInferenceModel._extract_text(response, "a/b") == "pong"
    )


def test_reasoning_model_content_is_preferred_over_scratchpad():
    response = {
        "choices": [
            {
                "message": {"content": "pong", "reasoning_content": "thinking..."},
                "finish_reason": "stop",
            }
        ]
    }
    assert HuggingFaceInferenceModel._extract_text(response, "a/b") == "pong"


def test_reasoning_truncation_raises_instead_of_returning_empty():
    """Observed live: content absent, reasoning_content present, length."""
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "reasoning_content": "thinking..."},
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 31}},
    }
    with pytest.raises(ReasoningTruncated) as excinfo:
        HuggingFaceInferenceModel._extract_text(response, "a/b")
    assert "max_tokens" in str(excinfo.value), "the error must name the fix"


def test_empty_response_without_reasoning_raises():
    response = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
    with pytest.raises(HuggingFaceModelError):
        HuggingFaceInferenceModel._extract_text(response, "a/b")


def test_missing_choices_raises():
    with pytest.raises(HuggingFaceModelError):
        HuggingFaceInferenceModel._extract_text({}, "a/b")


def test_default_max_tokens_is_large_enough_for_a_reasoning_model():
    """The budget must let a reasoning model finish thinking and still answer."""
    assert DEFAULT_MAX_TOKENS >= 1024


def test_unavailable_loading_and_truncated_are_all_huggingface_errors():
    """Callers that only care 'did it fail' must still catch them."""
    assert issubclass(ModelUnavailable, HuggingFaceModelError)
    assert issubclass(ModelLoading, ModelUnavailable)
    assert issubclass(ReasoningTruncated, HuggingFaceModelError)
    assert issubclass(RateLimited, HuggingFaceModelError)
    assert issubclass(PaymentRequired, HuggingFaceModelError)


# ---------------------------------------------------------------------------
# Error classification -- which failures are transient and model-specific
# ---------------------------------------------------------------------------

class _FakeResponse:
    """A stand-in for an aiohttp response context manager."""

    def __init__(self, status, body, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    """Serves canned responses (or raises) from ``model._post``'s level down."""

    def __init__(self, *script):
        # Each entry is a _FakeResponse or an Exception to raise.
        self._script = list(script)
        self.closed = False

    def post(self, url, json=None):
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.closed = True


def _model_with_session(session):
    model = _free_model()

    async def get_session():
        return session

    model._get_session = get_session
    return model


def test_503_model_loading_is_transient_and_carries_the_estimate():
    body = json.dumps(
        {"error": "Model a/b is currently loading", "estimated_time": 42.5}
    )
    model = _model_with_session(_FakeSession(_FakeResponse(503, body)))

    with pytest.raises(ModelLoading) as excinfo:
        asyncio.run(model.generate("hi"))
    assert excinfo.value.estimated_seconds == pytest.approx(42.5)


def test_503_without_a_loading_marker_is_a_plain_outage():
    model = _model_with_session(_FakeSession(_FakeResponse(503, "upstream error")))

    with pytest.raises(ModelUnavailable) as excinfo:
        asyncio.run(model.generate("hi"))
    assert not isinstance(excinfo.value, ModelLoading)


def test_502_provider_error_is_an_outage():
    body = json.dumps({"error": {"message": "provider error", "type": "x"}})
    model = _model_with_session(_FakeSession(_FakeResponse(502, body)))

    with pytest.raises(ModelUnavailable):
        asyncio.run(model.generate("hi"))


def test_429_is_rate_limited_and_reads_retry_after():
    model = _model_with_session(
        _FakeSession(_FakeResponse(429, "too many requests", {"Retry-After": "17"}))
    )

    with pytest.raises(RateLimited) as excinfo:
        asyncio.run(model.generate("hi"))
    assert excinfo.value.retry_after == pytest.approx(17.0)


def test_429_without_a_header_has_no_estimate():
    model = _model_with_session(_FakeSession(_FakeResponse(429, "slow down")))

    with pytest.raises(RateLimited) as excinfo:
        asyncio.run(model.generate("hi"))
    assert excinfo.value.retry_after is None


def test_402_is_payment_required_not_an_outage():
    """Observed live: depleted monthly credits return HTTP 402. Account-level,
    so it must not be mistaken for a flapping endpoint or a bad request."""
    body = json.dumps(
        {"error": "You have depleted your monthly included credits. Purchase "
                  "pre-paid credits to continue using Inference Providers."}
    )
    model = _model_with_session(_FakeSession(_FakeResponse(402, body)))

    with pytest.raises(PaymentRequired) as excinfo:
        asyncio.run(model.generate("hi"))
    assert not isinstance(excinfo.value, (ModelUnavailable, RateLimited))


def test_401_is_a_credential_error_not_a_transient_outage():
    """An invalid token must not be mistaken for a flapping endpoint."""
    body = json.dumps({"error": "Invalid username or password."})
    model = _model_with_session(_FakeSession(_FakeResponse(401, body)))

    with pytest.raises(HuggingFaceModelError) as excinfo:
        asyncio.run(model.generate("hi"))
    assert not isinstance(excinfo.value, (ModelUnavailable, RateLimited))


def test_400_is_a_request_error_not_an_outage():
    body = json.dumps({"error": {"message": "bad request"}})
    model = _model_with_session(_FakeSession(_FakeResponse(400, body)))

    with pytest.raises(HuggingFaceModelError) as excinfo:
        asyncio.run(model.generate("hi"))
    assert not isinstance(excinfo.value, ModelUnavailable)


def test_transport_errors_are_retried_then_raised():
    import aiohttp

    model = _model_with_session(
        _FakeSession(aiohttp.ClientError("reset"), aiohttp.ClientError("reset"))
    )
    model._max_retries = 1
    model._retry_delay = 0

    with pytest.raises(HuggingFaceModelError, match="after 2 attempts"):
        asyncio.run(model.generate("hi"))


def test_a_transport_error_then_success_is_a_retry_not_a_failure():
    import aiohttp

    ok = _FakeResponse(
        200, json.dumps({"choices": [{"message": {"content": "pong"}}]})
    )
    model = _model_with_session(_FakeSession(aiohttp.ClientError("blip"), ok))
    model._retry_delay = 0

    assert asyncio.run(model.generate("hi")) == "pong"


def test_model_outages_are_not_retried_in_place():
    """A downed backend stays down for minutes; generate_free moves on instead."""
    first = _FakeResponse(502, "provider error")
    second = _FakeResponse(
        200, json.dumps({"choices": [{"message": {"content": "pong"}}]})
    )
    session = _FakeSession(first, second)
    model = _model_with_session(session)
    model._retry_delay = 0

    with pytest.raises(ModelUnavailable):
        asyncio.run(model.generate("hi"))
    assert session._script == [second], "the second attempt must never happen"


def test_non_json_success_body_is_reported_not_misparsed():
    model = _model_with_session(
        _FakeSession(_FakeResponse(200, "<html>maintenance</html>"))
    )

    with pytest.raises(HuggingFaceModelError, match="non-JSON"):
        asyncio.run(model.generate("hi"))


def test_error_bodies_cannot_inject_newlines_into_logs():
    body = 'line one\r\nline two "quoted"\nline three'
    model = _model_with_session(_FakeSession(_FakeResponse(400, body)))

    with pytest.raises(HuggingFaceModelError) as excinfo:
        asyncio.run(model.generate("hi"))
    message = str(excinfo.value)
    assert "\n" not in message and "\r" not in message


def test_error_bodies_are_truncated():
    model = _model_with_session(_FakeSession(_FakeResponse(400, "x" * 5000)))

    with pytest.raises(HuggingFaceModelError) as excinfo:
        asyncio.run(model.generate("hi"))
    assert len(str(excinfo.value)) < 1000


# ---------------------------------------------------------------------------
# Request-body integrity -- the cost gate is only worth as much as the field
# it checked, so `model` must survive to the wire unchanged
# ---------------------------------------------------------------------------

def _free_model_recording_its_payload(**kwargs):
    model = _free_model(**kwargs)
    sent = {}

    async def record(path, payload):
        sent.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    model._post = record
    return model, sent


@pytest.mark.parametrize("field", ["model", "messages", "stream"])
def test_reserved_request_fields_cannot_be_overridden(field):
    """Overriding `model` would swap an approved free model for a paid one."""
    model, _ = _free_model_recording_its_payload()
    with pytest.raises(ReservedRequestField, match=field):
        asyncio.run(model.generate("hi", **{field: "paid/model"}))


def test_the_checked_model_is_the_model_actually_sent():
    """The positive half: what the policy approved is what goes on the wire."""
    model, sent = _free_model_recording_its_payload()
    asyncio.run(model.generate("hi"))
    assert sent["model"] == model.name


def test_a_free_model_is_pinned_to_its_free_provider_on_the_wire():
    """Without the pin the router picks :fastest, which may bill."""
    model, sent = _free_model_recording_its_payload(route="together")
    asyncio.run(model.generate("hi"))
    assert sent["model"] == "prism-ml/Ternary-Bonsai-27B-gguf:together"


def test_a_model_without_a_route_sends_its_bare_id():
    model, sent = _free_model_recording_its_payload()
    asyncio.run(model.generate("hi"))
    assert ":" not in sent["model"]


def test_ordinary_sampling_kwargs_are_still_forwarded():
    model, sent = _free_model_recording_its_payload()
    asyncio.run(model.generate("hi", top_p=0.9, frequency_penalty=0.5))
    assert sent["top_p"] == 0.9
    assert sent["frequency_penalty"] == 0.5
    assert sent["model"] == model.name, "controlled fields still win"


def test_system_prompt_becomes_a_system_message():
    model, sent = _free_model_recording_its_payload()
    asyncio.run(model.generate("hi", system_prompt="be terse"))
    assert sent["messages"][0] == {"role": "system", "content": "be terse"}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}


def test_structured_generation_also_refuses_a_model_override():
    model, _ = _free_model_recording_its_payload()
    with pytest.raises(ReservedRequestField):
        asyncio.run(
            model.generate_structured(
                "hi", schema={"type": "object"}, model="paid/model"
            )
        )


# ---------------------------------------------------------------------------
# Structured output must match the schema it asked for
# ---------------------------------------------------------------------------

def _model_replying(text):
    model = _free_model()

    async def reply(path, payload):
        return {"choices": [{"message": {"content": text}}]}

    model._post = reply
    return model


def test_structured_output_matching_the_schema_is_returned():
    model = _model_replying('{"city": "Paris", "country": "France"}')
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
    }
    result = asyncio.run(model.generate_structured("capital of France", schema))
    assert result == {"city": "Paris", "country": "France"}


def test_structured_output_missing_a_required_key_is_rejected():
    model = _model_replying('{"city": "Paris"}')
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
    }
    with pytest.raises(HuggingFaceModelError, match="schema"):
        asyncio.run(model.generate_structured("capital of France", schema))


def test_structured_output_that_is_not_json_is_rejected():
    model = _model_replying("I cannot help with that.")
    with pytest.raises(HuggingFaceModelError, match="valid JSON"):
        asyncio.run(model.generate_structured("hi", {"type": "object"}))


def test_a_fenced_reply_is_still_unwrapped_and_validated():
    model = _model_replying('```json\n{"a": 1}\n```')
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}},
        "required": ["a"],
    }
    assert asyncio.run(model.generate_structured("hi", schema)) == {"a": 1}


# ---------------------------------------------------------------------------
# Sessions -- a fallback chain must not pay a TLS handshake per attempt
# ---------------------------------------------------------------------------

def test_aclose_is_idempotent_and_safe_before_any_request():
    model = _free_model()

    async def close_twice():
        await model.aclose()
        await model.aclose()

    asyncio.run(close_twice())
    assert model._session is None


def test_session_is_reused_rather_than_rebuilt_per_request():
    """Constructing a session opens no connection, so this stays hermetic."""
    model = _free_model()

    async def run():
        first = await model._get_session()
        second = await model._get_session()
        assert first is second, "each request must not build a new session"
        assert not first.closed
        await model.aclose()
        return first

    session = asyncio.run(run())
    assert session.closed, "aclose() must actually close the session"


def test_async_context_manager_closes_a_real_session():
    model = _free_model()

    async def use():
        async with model as m:
            return await m._get_session()

    session = asyncio.run(use())
    assert session.closed, "leaving the context must release the connection"
    assert model._session is None


def test_session_carries_the_bearer_token_and_json_content_type():
    model = _free_model()

    async def run():
        session = await model._get_session()
        headers = session.headers
        await model.aclose()
        return headers

    headers = asyncio.run(run())
    assert headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Endpoint safety -- every request carries the bearer token
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://evil.example/v1",            # plaintext to a remote host
        "http://router.huggingface.co/v1",   # a plausible typo of the real URL
        "ftp://router.huggingface.co/v1",
        "router.huggingface.co/v1",          # no scheme at all
        "https://",                          # no host
    ],
)
def test_unsafe_base_urls_are_refused(url):
    with pytest.raises(InsecureEndpoint):
        validate_base_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://router.huggingface.co/v1",
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000",
        "http://[::1]:8000",
    ],
)
def test_https_and_loopback_are_accepted(url):
    """Loopback stays usable so a local mock router remains testable."""
    assert validate_base_url(url) == url.rstrip("/")


def test_trailing_slash_is_normalised():
    assert validate_base_url("https://router.huggingface.co/v1/") == (
        "https://router.huggingface.co/v1"
    )


def test_model_refuses_to_construct_against_a_plaintext_endpoint():
    """The check must run at construction, before any token is sent."""
    with pytest.raises(InsecureEndpoint):
        _free_model(base_url="http://evil.example/v1")


def test_provider_also_validates_its_endpoint():
    """The catalog fetch carries the token too, so it needs the same check."""
    from orchestrator.models.providers.base import ProviderConfig
    from orchestrator.models.providers.huggingface_provider import (
        HuggingFaceProvider,
    )

    with pytest.raises(InsecureEndpoint):
        HuggingFaceProvider(
            ProviderConfig(name="huggingface", base_url="http://evil.example/v1")
        )


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------

def _stuffed_provider(catalog):
    """A provider with a hand-loaded catalog and no network."""
    from orchestrator.models.providers.base import ProviderConfig
    from orchestrator.models.providers.huggingface_provider import (
        HuggingFaceProvider,
        model_cost_from_catalog,
    )

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.config = ProviderConfig(name="huggingface", api_key=FAKE_KEY)
    provider._base_url = "http://localhost:9/v1"  # loopback; never contacted
    provider._catalog = catalog
    provider._costs = {m: model_cost_from_catalog(e) for m, e in catalog.items()}
    return provider


def test_provider_transport_config_reaches_the_model():
    """Regression guard: the config must not be silently ignored."""
    from orchestrator.models.providers.base import ProviderConfig
    from orchestrator.models.providers.huggingface_provider import (
        HuggingFaceProvider,
    )

    provider = HuggingFaceProvider(
        ProviderConfig(
            name="huggingface", api_key=FAKE_KEY, timeout=12.5,
            max_retries=7, retry_delay=0.25,
        )
    )
    entry = _entry("a/b", [_provider("x", pricing={"input": 0, "output": 0})])
    provider._catalog = {"a/b": entry}
    provider._costs = {"a/b": ModelCost(is_free=True)}

    model = asyncio.run(provider.create_model("a/b"))
    assert model._timeout == 12.5
    assert model._max_retries == 7
    assert model._retry_delay == 0.25


def test_default_provider_timeout_suits_generation_not_metadata():
    """ProviderConfig's 30s default would cut off a slow generation."""
    from orchestrator.models.providers.huggingface_provider import (
        HuggingFaceProvider,
    )

    provider = HuggingFaceProvider()
    assert provider.config.timeout == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert provider.config.timeout > 30.0


def test_create_model_unknown_id_names_what_is_available():
    provider = _stuffed_provider(
        {"a/b": _entry("a/b", [_provider("x", pricing={"input": 0, "output": 0})])}
    )
    with pytest.raises(HuggingFaceModelError, match="a/b"):
        asyncio.run(provider.create_model("not/there"))


def test_create_model_pins_the_free_route():
    provider = _stuffed_provider(
        {
            "a/b": _entry(
                "a/b",
                [
                    _provider("novita", pricing={"input": 1, "output": 1}),
                    _provider("together", pricing={"input": 0, "output": 0}),
                ],
            )
        }
    )
    model = asyncio.run(provider.create_model("a/b"))
    assert model.route == "together"


def test_create_model_for_an_unpriced_entry_marks_pricing_unknown(monkeypatch):
    """Unknown price must not become a zero-filled, confidently wrong cost."""
    monkeypatch.setenv("ORCHESTRATOR_ALLOW_PAID_MODELS", "1")
    provider = _stuffed_provider(
        {"a/b": _entry("a/b", [_provider("novita")])}  # live but unpriced
    )

    model = asyncio.run(provider.create_model("a/b"))

    assert not model._pricing_is_known
    with pytest.raises(HuggingFaceModelError, match="without pricing"):
        asyncio.run(model.estimate_cost("hello"))


def test_create_model_for_an_unpriced_entry_is_refused_without_optin():
    provider = _stuffed_provider({"a/b": _entry("a/b", [_provider("novita")])})
    with pytest.raises(PaidModelRefused):
        asyncio.run(provider.create_model("a/b"))


def test_free_preference_orders_by_probed_throughput_and_drops_nothing():
    """No hard-coded model list: the catalog's probe data ranks the free set."""
    provider = _stuffed_provider(
        {
            "slow/model": _entry(
                "slow/model",
                [_provider("x", pricing={"input": 0, "output": 0}, throughput=10.0)],
            ),
            "fast/model": _entry(
                "fast/model",
                [_provider("y", pricing={"input": 0, "output": 0}, throughput=99.0)],
            ),
            "unprobed/model": _entry(
                "unprobed/model", [_provider("z", pricing={"input": 0, "output": 0})]
            ),
        }
    )

    ordered = provider.free_models_by_preference()

    assert ordered[0] == "fast/model"
    assert ordered[-1] == "unprobed/model", "unprobed models sort last"
    assert set(ordered) == set(provider._catalog), "no model may be dropped"


def test_capabilities_come_from_the_catalog_entry():
    provider = _stuffed_provider(
        {
            "v/m": _entry(
                "v/m",
                [
                    _provider("x", pricing={"input": 0, "output": 0}),
                ],
                modalities=("text", "image"),
            )
        }
    )
    provider._catalog["v/m"]["providers"][0]["context_length"] = 131072
    provider._catalog["v/m"]["providers"][0]["supports_structured_output"] = True

    caps = provider.get_model_capabilities("v/m")
    assert isinstance(caps, ModelCapabilities)
    assert caps.vision_capable
    assert caps.context_window == 131072
    assert caps.supports_structured_output


def test_generate_free_falls_back_past_loading_models():
    """A cold-starting model must not strand the caller."""
    from orchestrator.models.providers.huggingface_provider import (
        HuggingFaceProvider,
    )

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider._catalog = {"a/loading": {}, "b/ready": {}}
    provider._costs = {m: ModelCost(is_free=True) for m in provider._catalog}
    built = []

    async def fake_create(model_id, **kwargs):
        model = HuggingFaceInferenceModel(
            name=model_id, api_key=FAKE_KEY, cost=ModelCost(is_free=True)
        )
        if model_id == "a/loading":
            async def loading(path, payload):
                raise ModelLoading("loading", estimated_seconds=30.0)

            model._post = loading
        else:
            async def ok(path, payload):
                return {"choices": [{"message": {"content": "pong"}}]}

            model._post = ok
        built.append(model)
        return model

    provider.create_model = fake_create

    text, model_id = asyncio.run(provider.generate_free("hi"))
    assert text == "pong"
    assert model_id == "b/ready"
    assert all(m._session is None for m in built), "every attempt must be closed"


def test_generate_free_moves_past_a_truncated_reasoning_model():
    """A reasoning model that spent its budget thinking is not an answer."""
    from orchestrator.models.providers.huggingface_provider import (
        HuggingFaceProvider,
    )

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider._catalog = {"a/thinker": {}, "b/plain": {}}
    provider._costs = {m: ModelCost(is_free=True) for m in provider._catalog}

    async def fake_create(model_id, **kwargs):
        model = HuggingFaceInferenceModel(
            name=model_id, api_key=FAKE_KEY, cost=ModelCost(is_free=True)
        )

        async def truncated(path, payload):
            raise ReasoningTruncated("spent the budget thinking")

        async def ok(path, payload):
            return {"choices": [{"message": {"content": "pong"}}]}

        model._post = truncated if model_id == "a/thinker" else ok
        return model

    provider.create_model = fake_create

    text, model_id = asyncio.run(provider.generate_free("hi"))
    assert (text, model_id) == ("pong", "b/plain")


@pytest.mark.parametrize(
    "account_level_error",
    [
        RateLimited("slow down", retry_after=60.0),
        PaymentRequired("monthly credits depleted"),
    ],
)
def test_generate_free_does_not_walk_past_account_level_errors(
    account_level_error,
):
    """429/402 are account-level: trying every other model just hammers the
    same quota or bills the same empty balance."""
    from orchestrator.models.providers.huggingface_provider import (
        HuggingFaceProvider,
    )

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider._catalog = {"a/limited": {}, "b/never-tried": {}}
    provider._costs = {m: ModelCost(is_free=True) for m in provider._catalog}
    tried = []

    async def fake_create(model_id, **kwargs):
        tried.append(model_id)
        model = HuggingFaceInferenceModel(
            name=model_id, api_key=FAKE_KEY, cost=ModelCost(is_free=True)
        )

        async def limited(path, payload):
            raise account_level_error

        model._post = limited
        return model

    provider.create_model = fake_create

    with pytest.raises(type(account_level_error)):
        asyncio.run(provider.generate_free("hi"))
    assert tried == ["a/limited"], "no second model may be attempted"


def test_generate_free_raises_when_every_candidate_is_down():
    from orchestrator.models.providers.huggingface_provider import (
        HuggingFaceProvider,
    )

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider._catalog = {"a/down": {}, "b/down": {}}
    provider._costs = {m: ModelCost(is_free=True) for m in provider._catalog}

    async def fake_create(model_id, **kwargs):
        model = HuggingFaceInferenceModel(
            name=model_id, api_key=FAKE_KEY, cost=ModelCost(is_free=True)
        )

        async def down(path, payload):
            raise ModelUnavailable(f"{model_id} backend is down")

        model._post = down
        return model

    provider.create_model = fake_create

    with pytest.raises(HuggingFaceModelError, match="no free HuggingFace model"):
        asyncio.run(provider.generate_free("hi"))


def test_generate_free_with_an_empty_free_set_raises():
    from orchestrator.models.providers.huggingface_provider import (
        HuggingFaceProvider,
    )

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider._catalog = {}
    provider._costs = {}

    with pytest.raises(HuggingFaceModelError):
        asyncio.run(provider.generate_free("hi"))


# ---------------------------------------------------------------------------
# Registry integration -- only free models may be registered
# ---------------------------------------------------------------------------

def _seal_hf_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "orchestrator.models.huggingface_credentials._ORCHESTRATOR_ENV_FILE",
        tmp_path / "nope.env",
    )
    monkeypatch.setattr(
        "orchestrator.models.huggingface_credentials._HF_CLI_TOKEN_FILE",
        tmp_path / "nope-token",
    )


def test_hf_registry_population_is_skipped_without_a_credential(
    monkeypatch, tmp_path
):
    """No credential means no HuggingFace models -- and no network call."""
    from orchestrator._api import _register_free_huggingface_models
    from orchestrator.models.model_registry import ModelRegistry

    _seal_hf_credentials(monkeypatch, tmp_path)

    registry = ModelRegistry()
    assert _register_free_huggingface_models(registry) == 0
    assert registry.list_models() == []


def test_hf_registry_population_survives_an_unreachable_router(monkeypatch):
    """An outage must degrade to 'no HuggingFace models', not break startup."""
    from orchestrator import _api
    from orchestrator.models.model_registry import ModelRegistry

    monkeypatch.setenv("HF_TOKEN", FAKE_KEY)

    def unreachable(*args, **kwargs):
        raise OSError("Name or service not known")

    monkeypatch.setattr(
        "orchestrator.models.providers.huggingface_provider.fetch_catalog_sync",
        unreachable,
    )

    registry = ModelRegistry()
    assert _api._register_free_huggingface_models(registry) == 0


def test_hf_registry_population_registers_only_free_models(monkeypatch):
    from orchestrator import _api
    from orchestrator.models.model_registry import ModelRegistry

    monkeypatch.setenv("HF_TOKEN", FAKE_KEY)
    catalog = {
        "free/zero": _entry(
            "free/zero", [_provider("together", pricing={"input": 0, "output": 0})]
        ),
        "paid/model": _entry(
            "paid/model", [_provider("novita", pricing={"input": 1, "output": 2})]
        ),
    }
    monkeypatch.setattr(
        "orchestrator.models.providers.huggingface_provider.fetch_catalog_sync",
        lambda *args, **kwargs: catalog,
    )

    registry = ModelRegistry()
    registered = _api._register_free_huggingface_models(registry)

    assert registered == 1
    assert registry.list_models() == ["huggingface:free/zero"]
    model = registry.models["huggingface:free/zero"]
    assert model.cost.is_free
    assert model.route == "together", "the registered model keeps its free pin"

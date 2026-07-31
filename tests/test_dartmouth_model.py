"""Hermetic tests for the Dartmouth Chat adapter's decision logic.

These cover the parts that must be right *before* any network call: which
models count as free, whether a paid model is refused, and how a reasoning
model's truncated reply is reported. No mocks and no network -- every test
here exercises real functions over real catalog-shaped data.

The live round trip is covered by tests/test_live_dartmouth.py.
"""

import asyncio

import pytest

from orchestrator.core.model import ModelCost
from orchestrator.models.dartmouth_model import (
    ALLOW_PAID_ENV_VAR,
    DEFAULT_MAX_TOKENS,
    RESERVED_REQUEST_FIELDS,
    DartmouthModel,
    DartmouthModelError,
    InsecureEndpoint,
    PaidModelRefused,
    ReservedRequestField,
    paid_models_allowed,
    validate_base_url,
)
from orchestrator.models.providers.dartmouth_provider import model_cost_from_catalog

pytestmark = pytest.mark.unit

FAKE_KEY = "sk-0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _no_paid_optin(monkeypatch):
    monkeypatch.delenv(ALLOW_PAID_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# Free/paid classification -- this decides whether money gets spent
# ---------------------------------------------------------------------------

def test_zero_priced_model_is_free():
    """Shape taken from a real chat.dartmouth.edu catalog entry."""
    entry = {
        "id": "meta.llama-3-2-3b-instruct",
        "model_info": {"input_cost_per_token": 0, "output_cost_per_token": 0},
    }
    cost = model_cost_from_catalog(entry)
    assert cost.is_free is True
    assert cost.calculate_cost(1_000_000, 1_000_000) == 0.0


def test_priced_model_is_not_free_and_scales_per_1k():
    entry = {
        "id": "anthropic.claude-opus-5",
        "model_info": {
            "input_cost_per_token": 5e-06,
            "output_cost_per_token": 2.5e-05,
        },
    }
    cost = model_cost_from_catalog(entry)
    assert cost.is_free is False
    assert cost.input_cost_per_1k_tokens == pytest.approx(0.005)
    assert cost.output_cost_per_1k_tokens == pytest.approx(0.025)


def test_pricing_is_found_however_deeply_nested():
    """External models nest pricing a level deeper than internal ones."""
    entry = {
        "id": "vendor.model",
        "upstream_model_info": {
            "model_info": {
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
            }
        },
    }
    cost = model_cost_from_catalog(entry)
    assert cost.is_free is False
    assert cost.input_cost_per_1k_tokens == pytest.approx(0.001)


def test_unpriced_entry_is_treated_as_paid():
    """Absence of a price is not evidence of a zero price.

    Embeddings and helper bots carry no pricing. Assuming "free" for anything
    unpriced is the failure mode that quietly spends money, so these are
    excluded from the free set.
    """
    cost = model_cost_from_catalog({"id": "baai.bge-m3"})
    assert cost.is_free is False


def test_a_single_nonzero_price_makes_a_model_paid():
    """Free input with paid output is still paid."""
    entry = {
        "id": "vendor.mixed",
        "model_info": {"input_cost_per_token": 0, "output_cost_per_token": 3e-06},
    }
    assert model_cost_from_catalog(entry).is_free is False


# ---------------------------------------------------------------------------
# Paid-model gating
# ---------------------------------------------------------------------------

def test_free_model_constructs_without_optin():
    model = DartmouthModel(
        name="meta.llama-3-2-3b-instruct",
        api_key=FAKE_KEY,
        cost=ModelCost(is_free=True),
    )
    assert model.api_key_is_set is True


def test_paid_model_is_refused_by_default():
    """The refusal happens at construction, where the model was chosen."""
    with pytest.raises(PaidModelRefused) as excinfo:
        DartmouthModel(
            name="anthropic.claude-opus-5",
            api_key=FAKE_KEY,
            cost=ModelCost(
                input_cost_per_1k_tokens=0.005,
                output_cost_per_1k_tokens=0.025,
            ),
        )
    message = str(excinfo.value)
    assert ALLOW_PAID_ENV_VAR in message, "the error must say how to opt in"
    assert "0.005" in message, "the error must state what it would cost"


def test_paid_model_allowed_with_explicit_optin(monkeypatch):
    monkeypatch.setenv(ALLOW_PAID_ENV_VAR, "1")
    model = DartmouthModel(
        name="anthropic.claude-opus-5",
        api_key=FAKE_KEY,
        cost=ModelCost(input_cost_per_1k_tokens=0.005),
    )
    assert model.cost.is_free is False


@pytest.mark.parametrize("value", ["", "0", "no", "true", "yes", "TRUE"])
def test_only_the_exact_value_1_enables_paid_usage(monkeypatch, value):
    """Anything truthy-looking must not count -- spending needs an exact opt-in."""
    monkeypatch.setenv(ALLOW_PAID_ENV_VAR, value)
    assert paid_models_allowed() is False
    with pytest.raises(PaidModelRefused):
        DartmouthModel(
            name="anthropic.claude-opus-5",
            api_key=FAKE_KEY,
            cost=ModelCost(input_cost_per_1k_tokens=0.005),
        )


def test_unpriced_model_is_refused_rather_than_assumed_free():
    """Regression: `cost or ModelCost(is_free=True)` made this construct.

    A model built without consulting the catalog has an *unknown* price. The
    old default treated unknown as free, so
    `DartmouthModel(name="anthropic.claude-opus-5")` sailed past the cost gate
    and would have billed a real account on the first request.
    """
    with pytest.raises(PaidModelRefused) as excinfo:
        DartmouthModel(name="anthropic.claude-opus-5", api_key=FAKE_KEY)
    message = str(excinfo.value)
    assert "unknown" in message.lower(), "must say why it was refused"
    assert "create_model" in message, "must name the supported way to build it"


def test_unpriced_model_refuses_to_estimate_cost(monkeypatch):
    """A confidently wrong $0.00 budget is worse than an error."""
    monkeypatch.setenv(ALLOW_PAID_ENV_VAR, "1")
    model = DartmouthModel(name="anthropic.claude-opus-5", api_key=FAKE_KEY)
    with pytest.raises(DartmouthModelError, match="without pricing"):
        asyncio.run(model.estimate_cost("hello", max_tokens=100))


def test_estimate_cost_is_exactly_zero_for_free_models(anyio_backend=None):
    import asyncio

    model = DartmouthModel(
        name="meta.llama-3-2-3b-instruct",
        api_key=FAKE_KEY,
        cost=ModelCost(is_free=True),
    )
    assert asyncio.run(model.estimate_cost("x" * 10_000, max_tokens=8192)) == 0.0


# ---------------------------------------------------------------------------
# Response parsing -- the reasoning-model trap
# ---------------------------------------------------------------------------

def test_plain_content_is_returned():
    response = {"choices": [{"message": {"content": "pong"}}]}
    assert DartmouthModel._extract_text(response, "m") == "pong"


def test_reasoning_model_content_is_preferred_over_scratchpad():
    """A reasoning model returns both; the answer is `content`."""
    response = {
        "choices": [
            {
                "message": {
                    "content": "\n\npong",
                    "reasoning_content": "Thinking Process: ...",
                }
            }
        ]
    }
    assert DartmouthModel._extract_text(response, "m") == "\n\npong"


def test_reasoning_truncation_raises_instead_of_returning_empty():
    """Observed for real: qwen3.5-122b at max_tokens=16.

    It spent all 16 tokens thinking and returned content=null with 307
    completion tokens of reasoning. Returning "" would look like the model had
    nothing to say, when in fact the budget was too small -- a silent wrong
    answer instead of a fixable error.
    """
    response = {
        "choices": [
            {
                "message": {"content": None, "reasoning_content": "Thinking..."},
                "finish_reason": "length",
            }
        ]
    }
    with pytest.raises(DartmouthModelError) as excinfo:
        DartmouthModel._extract_text(response, "qwen.qwen3.5-122b")
    message = str(excinfo.value)
    assert "reasoning" in message.lower()
    assert "max_tokens" in message, "the error must name the fix"


def test_empty_response_without_reasoning_raises():
    response = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
    with pytest.raises(DartmouthModelError, match="empty response"):
        DartmouthModel._extract_text(response, "m")


def test_missing_choices_raises():
    with pytest.raises(DartmouthModelError, match="no choices"):
        DartmouthModel._extract_text({"choices": []}, "m")


def test_default_max_tokens_is_large_enough_for_a_reasoning_model():
    """A reasoning model observed using 307 tokens before answering.

    A small default silently truncates them, so this pins the headroom.
    """
    assert DEFAULT_MAX_TOKENS >= 1024


# ---------------------------------------------------------------------------
# Transient-outage classification -- free model endpoints flap independently
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "body",
    [
        # Verbatim from a real Dartmouth Chat 400 while qwen3.5 was down.
        '{"detail":"litellm.InternalServerError: InternalServerError: '
        "Hosted_vllmException - Cannot connect to host "
        'vllm-qwen35.ai-prod.svc.cluster.local:8000"}',
        '{"detail":"Available Model Group Fallbacks=None"}',
        "Service Unavailable",
        "model_not_loaded",
    ],
)
def test_downed_backend_is_recognised_as_transient(body):
    from orchestrator.models.dartmouth_model import _looks_unavailable

    assert _looks_unavailable(body) is True


@pytest.mark.parametrize(
    "body",
    [
        '{"detail":"invalid model id"}',
        '{"error":{"message":"max_tokens must be positive"}}',
        "unauthorized",
    ],
)
def test_genuine_request_errors_are_not_treated_as_outages(body):
    """Retrying a bad request against every model just wastes time."""
    from orchestrator.models.dartmouth_model import _looks_unavailable

    assert _looks_unavailable(body) is False


def test_reasoning_truncation_has_its_own_type():
    """`generate_free` falls back on this but not on a generic failure."""
    from orchestrator.models.dartmouth_model import ReasoningTruncated

    response = {
        "choices": [
            {
                "message": {"content": None, "reasoning_content": "..."},
                "finish_reason": "length",
            }
        ]
    }
    with pytest.raises(ReasoningTruncated):
        DartmouthModel._extract_text(response, "qwen.qwen3.5-122b")


def test_unavailable_and_truncated_are_both_dartmouth_errors():
    """Callers that only care 'did it fail' must still catch them."""
    from orchestrator.models.dartmouth_model import ModelUnavailable, ReasoningTruncated

    assert issubclass(ModelUnavailable, DartmouthModelError)
    assert issubclass(ReasoningTruncated, DartmouthModelError)


# ---------------------------------------------------------------------------
# Request-body integrity -- the cost gate is only worth as much as the field
# it checked, so `model` must survive to the wire unchanged
# ---------------------------------------------------------------------------

def _free_model_recording_its_payload():
    """A free model whose transport records the request instead of sending it."""
    model = DartmouthModel(
        name="meta.llama-3-2-3b-instruct",
        api_key=FAKE_KEY,
        cost=ModelCost(is_free=True),
    )
    sent = {}

    async def record(path, payload):
        sent.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    model._post = record
    return model, sent


@pytest.mark.parametrize("field", sorted(RESERVED_REQUEST_FIELDS))
def test_reserved_request_fields_cannot_be_overridden(field):
    """Regression: `payload.update(kwargs)` let kwargs replace `model`.

    The free/paid decision is made at construction against `self.name`. When
    the request body could still be handed a different `model`, an approved
    free model became an *unapproved paid one* at call time -- the check
    passed and the bill arrived anyway.
    """
    model, _ = _free_model_recording_its_payload()
    with pytest.raises(ReservedRequestField, match=field):
        asyncio.run(model.generate("hi", **{field: "anthropic.claude-opus-5"}))


def test_the_checked_model_is_the_model_actually_sent():
    """The positive half: what the policy approved is what goes on the wire."""
    model, sent = _free_model_recording_its_payload()
    asyncio.run(model.generate("hi"))
    assert sent["model"] == model.name


def test_ordinary_sampling_kwargs_are_still_forwarded():
    """The guard must not turn into a blanket refusal of tuning parameters."""
    model, sent = _free_model_recording_its_payload()
    asyncio.run(model.generate("hi", top_p=0.9, frequency_penalty=0.5))
    assert sent["top_p"] == 0.9
    assert sent["frequency_penalty"] == 0.5
    assert sent["model"] == model.name, "controlled fields still win"


def test_structured_generation_also_refuses_a_model_override():
    """generate_structured() forwards kwargs too, so it inherits the guard."""
    model, _ = _free_model_recording_its_payload()
    with pytest.raises(ReservedRequestField):
        asyncio.run(
            model.generate_structured(
                "hi", schema={"type": "object"}, model="anthropic.claude-opus-5"
            )
        )


# ---------------------------------------------------------------------------
# Endpoint safety -- every request carries the bearer token
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://evil.example/api",       # plaintext to a remote host
        "http://chat.dartmouth.edu/api",  # a plausible typo of the real URL
        "ftp://chat.dartmouth.edu/api",
        "chat.dartmouth.edu/api",        # no scheme at all
        "https://",                      # no host
    ],
)
def test_unsafe_base_urls_are_refused(url):
    with pytest.raises(InsecureEndpoint):
        validate_base_url(url)


@pytest.mark.parametrize(
    "url",
    ["https://chat.dartmouth.edu/api", "http://localhost:8000/api",
     "http://127.0.0.1:8000", "http://[::1]:8000"],
)
def test_https_and_loopback_are_accepted(url):
    """Loopback stays usable so a local mock gateway remains testable."""
    assert validate_base_url(url) == url.rstrip("/")


def test_trailing_slash_is_normalised():
    assert validate_base_url("https://chat.dartmouth.edu/api/") == (
        "https://chat.dartmouth.edu/api"
    )


def test_model_refuses_to_construct_against_a_plaintext_endpoint():
    """The check must run at construction, before any token is sent."""
    with pytest.raises(InsecureEndpoint):
        DartmouthModel(
            name="meta.llama-3-2-3b-instruct",
            api_key=FAKE_KEY,
            base_url="http://evil.example/api",
            cost=ModelCost(is_free=True),
        )


def test_provider_also_validates_its_endpoint():
    """The catalog fetch carries the token too, so it needs the same check."""
    from orchestrator.models.providers.base import ProviderConfig
    from orchestrator.models.providers.dartmouth_provider import DartmouthProvider

    with pytest.raises(InsecureEndpoint):
        DartmouthProvider(
            ProviderConfig(name="dartmouth", base_url="http://evil.example/api")
        )


def test_free_preference_ranks_strongest_first_and_keeps_unknowns():
    """Ordering must not drop a model the catalog added but we don't know."""
    from orchestrator.models.providers.dartmouth_provider import DartmouthProvider

    provider = DartmouthProvider.__new__(DartmouthProvider)
    provider._catalog = {m: {} for m in [
        "meta.llama-3-2-3b-instruct", "qwen.qwen3.5-122b",
        "google.gemma-4-31B-it", "vendor.brand-new-model",
    ]}
    provider._costs = {m: ModelCost(is_free=True) for m in provider._catalog}

    ordered = provider.free_models_by_preference()

    assert ordered[0] == "qwen.qwen3.5-122b", "strongest model should be first"
    assert ordered[-1] == "vendor.brand-new-model", "unknown models sort last"
    assert set(ordered) == set(provider._catalog), "no model may be dropped"

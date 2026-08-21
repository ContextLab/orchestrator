"""HuggingFace Inference API model adapter.

The HuggingFace router (https://router.huggingface.co/v1) is an
OpenAI-compatible gateway onto the Inference Providers network: one token,
many backend providers, server-side routing. Deliberately implemented against
the HTTP API with ``aiohttp`` -- already a core dependency -- rather than
through ``huggingface_hub``. The wire format is a documented, stable
OpenAI-compatible contract, and adding an SDK to speak it would put a heavy
dependency on the core install path for no capability gain (ADR 0001's
dependency policy).

Free-first is enforced here rather than left to the caller. Unlike Dartmouth
Chat, nothing on this gateway is permanently zero-cost: a model is free only
while some live provider carries an ``is_free`` promo or explicit zero
pricing, and an *unpinned* request routes ``:fastest`` -- which may be a paid
provider. So a free model is pinned to its free provider on the wire
(``model_id:provider``), and a paid model is refused unless
:data:`ALLOW_PAID_ENV_VAR` is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from ..core.model import Model, ModelCapabilities, ModelCost, ModelRequirements
from .dartmouth_model import ALLOW_PAID_ENV_VAR, paid_models_allowed
from .huggingface_credentials import mask_key, resolve_huggingface_api_key

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOW_PAID_ENV_VAR",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "HuggingFaceInferenceModel",
    "HuggingFaceModelError",
    "InsecureEndpoint",
    "ModelLoading",
    "ModelUnavailable",
    "PaidModelRefused",
    "PaymentRequired",
    "RateLimited",
    "ReasoningTruncated",
    "ReservedRequestField",
    "validate_base_url",
]

#: Overridable for testing against a different router.
DEFAULT_BASE_URL = os.environ.get(
    "HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1"
)

#: Reasoning models spend tokens on `reasoning_content` *before* emitting any
#: `content`. With a small budget the whole allowance is consumed thinking and
#: `content` comes back absent -- observed live with
#: prism-ml/Ternary-Bonsai-27B-AWQ-4bit at max_tokens=32, which returned 31
#: reasoning tokens and no answer. This default is large enough that a
#: reasoning model can finish and still answer.
DEFAULT_MAX_TOKENS = 2048

#: Generation can legitimately take minutes when a model cold-starts behind
#: the router, so this is far longer than ``ProviderConfig.timeout``'s 30s
#: default. A provider built with an explicit config uses that config's value.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300

#: Router error bodies are echoed into exceptions and logs for diagnosis.
#: They are attacker-influenced (a prompt can be reflected back), so they are
#: truncated and stripped of control characters first -- an unescaped newline
#: or carriage return lets one line of response forge additional log lines.
_MAX_ERROR_BODY_CHARS = 500


def _safe_error_body(body: str) -> str:
    """Render a router error body safely for an exception or log line."""
    collapsed = " ".join(body.split())
    if len(collapsed) > _MAX_ERROR_BODY_CHARS:
        collapsed = collapsed[:_MAX_ERROR_BODY_CHARS] + "... (truncated)"
    return collapsed


class HuggingFaceModelError(RuntimeError):
    """Raised when a HuggingFace Inference API request fails."""


class PaidModelRefused(HuggingFaceModelError):
    """Raised when a paid model is requested without an explicit opt-in."""


class ModelUnavailable(HuggingFaceModelError):
    """Raised when the router is up but this model's providers are not.

    Distinct from :class:`HuggingFaceModelError` because it is transient and
    model-specific: the router returns 502 "provider error" when every
    provider serving a model is failing. A caller seeing this should try a
    different model, not give up.
    """


class ModelLoading(ModelUnavailable):
    """Raised when the model is cold-starting behind the router (HTTP 503).

    Carries the router's ``estimated_time`` when present so a caller can
    decide whether to wait or move on; :meth:`HuggingFaceProvider.generate_free`
    moves on.
    """

    def __init__(self, message: str, *, estimated_seconds: Optional[float] = None):
        super().__init__(message)
        self.estimated_seconds = estimated_seconds


class RateLimited(HuggingFaceModelError):
    """Raised on HTTP 429. Account-level, not model-specific.

    Walking a fallback chain of other models does not help -- every request
    draws on the same account quota -- so ``generate_free`` deliberately lets
    this propagate rather than hammering the API.
    """

    def __init__(self, message: str, *, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class PaymentRequired(HuggingFaceModelError):
    """Raised on HTTP 402: the account's included monthly credits are gone.

    Observed live (2026-08-21): the router answers ``402`` with "You have
    depleted your monthly included credits". Account-level like
    :class:`RateLimited`, but not transient on any useful timescale -- the
    credits reset monthly -- so ``generate_free`` lets it propagate too.
    """


class ReservedRequestField(HuggingFaceModelError):
    """Raised when a caller tries to override a field this adapter controls.

    ``model`` is the field the free/paid policy was checked against at
    construction, so silently letting a request body override it converts an
    approved free model into an unapproved paid one *after* the check. That is
    a policy bypass, not a convenience, so it is refused rather than ignored.
    """


class InsecureEndpoint(HuggingFaceModelError):
    """Raised when a router URL would send the bearer token in the clear."""


class ReasoningTruncated(HuggingFaceModelError):
    """Raised when a reasoning model spent its whole budget thinking.

    Separate from a generic failure because it is *recoverable two ways*:
    raise ``max_tokens``, or ask a model that does not emit a reasoning
    scratchpad. :meth:`HuggingFaceProvider.generate_free` uses the second.
    """


#: Substrings in a router error body that mean "this model's backends are
#: down" rather than "your request was wrong".
_UNAVAILABLE_MARKERS = (
    "cannot connect to host",
    "model_not_loaded",
    "no healthy upstream",
    "provider error",
    "service unavailable",
    "temporarily unavailable",
)

#: Substrings that mean the 503 is a cold start, not an outage.
_LOADING_MARKERS = ("currently loading", "model is loading")


def _looks_unavailable(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _UNAVAILABLE_MARKERS)


def _parse_estimated_seconds(body: str) -> Optional[float]:
    """Pull ``estimated_time`` out of a model-loading body, when present."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(payload, dict):
        value = payload.get("estimated_time")
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    return None


def _parse_retry_after(headers: Any) -> Optional[float]:
    """Seconds until the rate limit resets, from ``Retry-After`` when sent."""
    value = (headers or {}).get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


#: Request fields this adapter owns. A caller may tune sampling, penalties and
#: the token budget, but not these -- see :class:`ReservedRequestField`.
#: ``stream`` is included because a streamed reply is a series of SSE events,
#: which ``_extract_text`` would misread as a malformed response.
RESERVED_REQUEST_FIELDS = frozenset({"model", "messages", "stream"})

#: Hosts allowed to serve the router over plaintext HTTP. Every request
#: carries the bearer token in an ``Authorization`` header, so anything that
#: leaves the machine must be TLS. Loopback is exempt so a local mock router
#: remains testable.
_PLAINTEXT_OK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def validate_base_url(url: str) -> str:
    """Return ``url`` normalised, or raise if it would leak the credential.

    A mistyped or hostile ``base_url`` receives the bearer token on the very
    first request, so the scheme is checked before any call is made rather
    than trusted.

    Raises:
        InsecureEndpoint: if the URL is malformed, or is plaintext HTTP to
        anything other than loopback.
    """
    cleaned = url.strip().rstrip("/")
    parts = urlsplit(cleaned)

    if not parts.scheme or not parts.netloc:
        raise InsecureEndpoint(
            f"HuggingFace base_url {url!r} is not a valid absolute URL. "
            f"Expected something like {DEFAULT_BASE_URL!r}."
        )
    if parts.scheme == "https":
        return cleaned
    if parts.scheme == "http" and (parts.hostname or "") in _PLAINTEXT_OK_HOSTS:
        return cleaned
    raise InsecureEndpoint(
        f"HuggingFace base_url {url!r} uses {parts.scheme!r}, which would send "
        f"the API token unencrypted to {parts.hostname!r}. Use https:// "
        f"(plaintext http:// is permitted only for "
        f"{', '.join(sorted(_PLAINTEXT_OK_HOSTS))})."
    )


class HuggingFaceInferenceModel(Model):
    """A chat model served through the HuggingFace Inference Providers router.

    Named to stay distinct from the retired local-transformers
    ``HuggingFaceModel`` adapter (#430): this class speaks only to the hosted
    router and never downloads weights.
    """

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        route: Optional[str] = None,
        capabilities: Optional[ModelCapabilities] = None,
        requirements: Optional[ModelRequirements] = None,
        cost: Optional[ModelCost] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            name: A Hub model id the router serves, e.g. ``openai/gpt-oss-120b``.
            api_key: Bearer token. Resolved from the environment or the local
                credential stores when omitted.
            base_url: Router root. Must be HTTPS unless it is loopback.
            route: Provider to pin on the wire (``name:route``). The provider
                attaches this for free models, where an unpinned request could
                route to a paid provider and bill the account.
            cost: Real pricing from the live catalog, normally supplied by
                :meth:`HuggingFaceProvider.create_model`. Omitting it means
                the price is **unknown**, which is treated as paid.

        Raises:
            PaidModelRefused: if the model costs money, or its price is
                unknown, and the paid opt-in is not set.
            InsecureEndpoint: if ``base_url`` would leak the credential.
        """
        # Distinguishes "the catalog says this is free" from "nobody asked the
        # catalog". Only the former is safe to run without an opt-in.
        self._pricing_is_known = cost is not None

        super().__init__(
            name=name,
            provider="huggingface",
            # ModelCapabilities requires at least one task, so a bare
            # ModelCapabilities() is not constructible. The provider supplies
            # richer capabilities from the catalog; this is the floor for a
            # model built directly.
            capabilities=capabilities
            or ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform", "summarize"],
                supports_structured_output=True,
            ),
            requirements=requirements or ModelRequirements(),
            # NOT `cost or ModelCost(is_free=True)`. Defaulting an unpriced
            # model to free would let a typo'd model id skip the cost gate
            # entirely and bill a real account.
            cost=cost if cost is not None else ModelCost(is_free=False),
            **kwargs,
        )

        credential = (
            None if api_key else resolve_huggingface_api_key(required=True)
        )
        self._api_key = api_key or (credential.key if credential else "")
        self._base_url = validate_base_url(base_url or DEFAULT_BASE_URL)
        self.route = route
        self._is_available = True
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        # Created on first request and reused, so a fallback chain walking
        # several models does not pay a fresh TLS handshake each time. Closed
        # by aclose(); see the async-context-manager support below.
        self._session: Optional[Any] = None

        if credential is not None:
            logger.debug(
                "HuggingFace credential %s resolved from %s",
                mask_key(self._api_key),
                credential.source,
            )

        self._enforce_cost_policy()

    def _enforce_cost_policy(self) -> None:
        """Refuse a paid model unless the operator opted in.

        Checked at construction rather than at call time so the failure lands
        where the model was chosen, not deep inside a pipeline run.

        Unknown pricing is refused alongside known-paid pricing. The router
        catalog is the only authority on what a model costs, so a model built
        without consulting it has an unknown price -- and treating unknown as
        free is precisely the assumption that spends money by accident.
        """
        if paid_models_allowed():
            return
        if not self._pricing_is_known:
            raise PaidModelRefused(
                f"{self.name!r} was constructed without pricing, so its cost "
                f"is unknown and it is treated as paid. Build it through "
                f"HuggingFaceProvider.create_model(), which attaches real "
                f"pricing from the live catalog, or pass an explicit cost=. "
                f"Set {ALLOW_PAID_ENV_VAR}=1 to permit unpriced and paid usage."
            )
        if self.cost.is_free:
            return
        raise PaidModelRefused(
            f"{self.name!r} costs money "
            f"(input ${self.cost.input_cost_per_1k_tokens:.6f}/1k, "
            f"output ${self.cost.output_cost_per_1k_tokens:.6f}/1k) and "
            f"{ALLOW_PAID_ENV_VAR} is not set to '1'. The router catalog "
            f"marks free routes -- see HuggingFaceProvider.list_free_models() "
            f"-- or set {ALLOW_PAID_ENV_VAR}=1 to permit paid usage."
        )

    @property
    def api_key_is_set(self) -> bool:
        """Whether a credential is available. Never exposes the token itself."""
        return bool(self._api_key)

    @property
    def _wire_model_id(self) -> str:
        """The model id sent on the wire, with any provider pin applied."""
        return f"{self.name}:{self.route}" if self.route else self.name

    async def _get_session(self) -> Any:
        """Return the shared HTTP session, opening it on first use."""
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def aclose(self) -> None:
        """Close the shared HTTP session. Safe to call more than once."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "HuggingFaceInferenceModel":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON to the router and return the decoded response.

        Transport failures (connection reset, DNS blip) are retried
        ``max_retries`` times. A *model* being down, cold-starting, or the
        account being rate-limited is deliberately NOT retried in place:
        cold starts and outages outlast any sensible retry loop, and a 429 is
        account-level, so retrying just hammers the same quota. Callers with a
        fallback chain (:meth:`HuggingFaceProvider.generate_free`) move to the
        next model instead.
        """
        import aiohttp

        url = f"{self._base_url}/{path.lstrip('/')}"
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload) as response:
                    body = await response.text()
                    if response.status >= 400:
                        # The body may echo the request but never the bearer
                        # token. It is still attacker-influenced, so it is
                        # sanitised before going into an exception or a log.
                        detail = (
                            f"HuggingFace router returned HTTP "
                            f"{response.status} for model {self.name!r}: "
                            f"{_safe_error_body(body)}"
                        )
                        if response.status == 429:
                            raise RateLimited(
                                detail,
                                retry_after=_parse_retry_after(response.headers),
                            )
                        if response.status == 402:
                            raise PaymentRequired(detail)
                        if response.status == 503 and any(
                            marker in body.lower() for marker in _LOADING_MARKERS
                        ):
                            raise ModelLoading(
                                detail,
                                estimated_seconds=_parse_estimated_seconds(body),
                            )
                        if response.status >= 500 or _looks_unavailable(body):
                            raise ModelUnavailable(detail)
                        raise HuggingFaceModelError(detail)
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError as exc:
                        # A maintenance window serves an HTML redirect page.
                        raise HuggingFaceModelError(
                            f"HuggingFace router returned a non-JSON response "
                            f"for {self.name!r} (is the router in "
                            f"maintenance?): {_safe_error_body(body)}"
                        ) from exc
            except aiohttp.ClientError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "HuggingFace transport error for %s (attempt %d/%d), "
                        "retrying: %s",
                        self.name,
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                    )
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                    # The session may be poisoned by the failure; drop it so
                    # the next attempt opens a fresh connection.
                    await self.aclose()

        raise HuggingFaceModelError(
            f"HuggingFace router request failed for {self.name!r} after "
            f"{self._max_retries + 1} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _extract_text(response: Dict[str, Any], model_name: str) -> str:
        """Pull the assistant text out of a chat-completion response.

        Reasoning models put their scratchpad in ``reasoning_content`` and the
        answer in ``content``. When the token budget is exhausted while
        thinking, ``content`` is absent -- which is a truncation, not an empty
        answer, and must not be returned as an empty string.
        """
        choices = response.get("choices") or []
        if not choices:
            raise HuggingFaceModelError(
                f"HuggingFace router returned no choices for {model_name!r}"
            )
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")

        if content:
            return content

        finish_reason = choice.get("finish_reason")
        if message.get("reasoning_content"):
            raise ReasoningTruncated(
                f"{model_name!r} produced only reasoning tokens and no answer "
                f"(finish_reason={finish_reason!r}). This is a reasoning "
                f"model: raise max_tokens (default {DEFAULT_MAX_TOKENS}) so it "
                f"can finish thinking and still reply."
            )
        raise HuggingFaceModelError(
            f"{model_name!r} returned an empty response "
            f"(finish_reason={finish_reason!r})"
        )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from ``prompt``.

        Extra ``kwargs`` are forwarded to the router as request fields, so
        sampling parameters work as expected. Fields in
        :data:`RESERVED_REQUEST_FIELDS` are refused rather than forwarded.

        Raises:
            ReservedRequestField: if ``kwargs`` would override a field this
                adapter controls -- notably ``model``, which the cost policy
                was checked against.
        """
        reserved = sorted(RESERVED_REQUEST_FIELDS.intersection(kwargs))
        if reserved:
            raise ReservedRequestField(
                f"cannot override {', '.join(repr(f) for f in reserved)} on a "
                f"request to {self.name!r}: these fields are set by the "
                f"adapter. Overriding 'model' in particular would bypass the "
                f"free/paid check already made for {self.name!r} -- construct "
                f"a different model instead."
            )

        messages = []
        system_prompt = kwargs.pop("system_prompt", None)
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Caller-supplied fields go in first and the controlled fields are
        # written over them, so the reserved-field check above is belt and
        # braces: even if it were bypassed, `model` still cannot be swapped.
        payload: Dict[str, Any] = dict(kwargs)
        payload.update(
            {
                "model": self._wire_model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            }
        )

        response = await self._post("chat/completions", payload)
        return self._extract_text(response, self.name)

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate JSON conforming to ``schema``.

        Only some providers behind the router honour a ``response_format``
        parameter (the catalog's ``supports_structured_output`` flag), so the
        schema is stated in the prompt instead and the reply is parsed. A
        reply that is not valid JSON raises rather than being silently coerced
        into a string.
        """
        instruction = (
            f"{prompt}\n\n"
            f"Respond with JSON only -- no prose, no markdown fences -- "
            f"conforming to this JSON Schema:\n{json.dumps(schema, indent=2)}"
        )
        text = await self.generate(instruction, temperature=temperature, **kwargs)

        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Models frequently fence JSON despite being told not to.
            cleaned = cleaned.split("```", 2)[1] if "```" in cleaned[3:] else cleaned
            cleaned = cleaned.removeprefix("json").strip().strip("`").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise HuggingFaceModelError(
                f"{self.name!r} did not return valid JSON: {exc}. "
                f"Response began: {text[:200]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise HuggingFaceModelError(
                f"{self.name!r} returned a {type(parsed).__name__}, expected a "
                f"JSON object"
            )

        # Parsing proves the reply is JSON, not that it is the JSON that was
        # asked for. Small models routinely return well-formed objects with
        # the wrong keys or types, which would otherwise flow downstream as if
        # the schema had been honoured.
        import jsonschema

        try:
            jsonschema.validate(instance=parsed, schema=schema)
        except jsonschema.ValidationError as exc:
            raise HuggingFaceModelError(
                f"{self.name!r} returned JSON that does not match the "
                f"requested schema at {list(exc.absolute_path) or '<root>'}: "
                f"{exc.message}. Received: {json.dumps(parsed)[:200]}"
            ) from exc
        except jsonschema.SchemaError as exc:
            raise HuggingFaceModelError(
                f"the schema passed to generate_structured() is itself "
                f"invalid: {exc.message}"
            ) from exc
        return parsed

    async def health_check(self) -> bool:
        """Whether the router will serve this model."""
        try:
            await self.generate("ping", temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS)
            return True
        except Exception as exc:  # noqa: BLE001 - health checks report, not raise
            logger.warning(
                "HuggingFace health check failed for %s: %s", self.name, exc
            )
            return False
        finally:
            # A health check is a one-shot probe, so it must not leave a
            # session open behind it.
            await self.aclose()

    async def estimate_cost(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> float:
        """Estimate USD cost. Exactly 0.0 for the free routes.

        Raises:
            HuggingFaceModelError: if the model was built without pricing. The
                zero-filled default would otherwise report $0.00 for a model
                that may well bill -- a confidently wrong budget number is
                worse than an error.
        """
        if not self._pricing_is_known:
            raise HuggingFaceModelError(
                f"cannot estimate cost for {self.name!r}: it was constructed "
                f"without pricing. Build it through "
                f"HuggingFaceProvider.create_model() to attach real pricing."
            )
        if self.cost.is_free:
            return 0.0
        # ~4 characters per token is the usual rough English estimate; this is
        # a budgeting aid, not billing.
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max_tokens or DEFAULT_MAX_TOKENS
        return self.cost.calculate_cost(input_tokens, output_tokens)

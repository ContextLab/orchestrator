"""Dartmouth Chat model adapter.

Dartmouth Chat (https://chat.dartmouth.edu/api/) is an OpenAI-compatible
gateway. Several of the models it serves cost **nothing per token**, so this
adapter is the cheapest way to run real models against this project.

Deliberately implemented against the HTTP API with ``aiohttp`` -- already a
core dependency -- rather than through ``openai`` or ``langchain-dartmouth``.
The wire format is a documented, stable OpenAI-compatible contract, and adding
an SDK to speak it would put a heavy dependency on the core install path for
no capability gain.

Free-first is enforced here rather than left to the caller: a paid model is
refused unless :data:`ALLOW_PAID_ENV_VAR` is set, so an accidental model-name
typo cannot quietly start spending money.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from ..core.model import Model, ModelCapabilities, ModelCost, ModelRequirements
from .dartmouth_credentials import mask_key, resolve_dartmouth_api_key

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOW_PAID_ENV_VAR",
    "DEFAULT_BASE_URL",
    "DartmouthModel",
    "DartmouthModelError",
    "InsecureEndpoint",
    "PaidModelRefused",
    "ReservedRequestField",
    "validate_base_url",
]

#: Overridable for testing against a different gateway.
DEFAULT_BASE_URL = os.environ.get(
    "DARTMOUTH_CHAT_BASE_URL", "https://chat.dartmouth.edu/api"
)

#: Set to "1" to permit models that cost money. Off by default.
ALLOW_PAID_ENV_VAR = "ORCHESTRATOR_ALLOW_PAID_MODELS"

#: Reasoning models spend tokens on `reasoning_content` *before* emitting any
#: `content`. With a small budget the whole allowance is consumed thinking and
#: `content` comes back null -- observed with qwen3.5-122b at max_tokens=16,
#: which returned 307 completion tokens of reasoning and no answer. This
#: default is large enough that a reasoning model can finish and still answer.
DEFAULT_MAX_TOKENS = 2048

#: Generation can legitimately take minutes on a busy shared cluster, so this
#: is far longer than ``ProviderConfig.timeout``'s 30s default. A provider
#: built with an explicit config uses that config's value instead.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300

#: Gateway error bodies are echoed into exceptions and logs for diagnosis.
#: They are attacker-influenced (a prompt can be reflected back), so they are
#: truncated and stripped of control characters first -- an unescaped newline
#: or carriage return lets one line of response forge additional log lines.
_MAX_ERROR_BODY_CHARS = 500


def _safe_error_body(body: str) -> str:
    """Render a gateway error body safely for an exception or log line."""
    collapsed = " ".join(body.split())
    if len(collapsed) > _MAX_ERROR_BODY_CHARS:
        collapsed = collapsed[:_MAX_ERROR_BODY_CHARS] + "... (truncated)"
    return collapsed


class DartmouthModelError(RuntimeError):
    """Raised when a Dartmouth Chat request fails."""


class PaidModelRefused(DartmouthModelError):
    """Raised when a paid model is requested without an explicit opt-in."""


class ModelUnavailable(DartmouthModelError):
    """Raised when the gateway is up but this model's backend is not.

    Distinct from :class:`DartmouthModelError` because it is transient and
    model-specific: the free models are served from individual cluster
    endpoints that go down independently, e.g.

        Cannot connect to host vllm-qwen35.ai-prod.svc.cluster.local:8000

    A caller seeing this should try a different model, not give up.
    """


class ReservedRequestField(DartmouthModelError):
    """Raised when a caller tries to override a field this adapter controls.

    ``model`` is the field the free/paid policy was checked against at
    construction, so silently letting a request body override it converts an
    approved free model into an unapproved paid one *after* the check. That is
    a policy bypass, not a convenience, so it is refused rather than ignored.
    """


class InsecureEndpoint(DartmouthModelError):
    """Raised when a gateway URL would send the bearer token in the clear."""


class ReasoningTruncated(DartmouthModelError):
    """Raised when a reasoning model spent its whole budget thinking.

    Separate from a generic failure because it is *recoverable two ways*:
    raise ``max_tokens``, or ask a model that does not emit a reasoning
    scratchpad. :meth:`DartmouthProvider.generate_free` uses the second.
    """


#: Substrings in a gateway error that mean "this model's backend is down"
#: rather than "your request was wrong".
_UNAVAILABLE_MARKERS = (
    "cannot connect to host",
    "internalservererror",
    "model group fallbacks=none",
    "service unavailable",
    "temporarily unavailable",
    "model_not_loaded",
)


def _looks_unavailable(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _UNAVAILABLE_MARKERS)


def paid_models_allowed() -> bool:
    """Whether models that cost money may be used."""
    return os.environ.get(ALLOW_PAID_ENV_VAR, "").strip() == "1"


#: Request fields this adapter owns. A caller may tune sampling, penalties and
#: the token budget, but not these -- see :class:`ReservedRequestField`.
#: ``stream`` is included because a streamed reply is a series of SSE events,
#: which :meth:`DartmouthModel._extract_text` would misread as a malformed
#: response rather than a stream.
RESERVED_REQUEST_FIELDS = frozenset({"model", "messages", "stream"})

#: Hosts allowed to serve the gateway over plaintext HTTP. Every request
#: carries the bearer token in an ``Authorization`` header, so anything that
#: leaves the machine must be TLS. Loopback is exempt so a local mock gateway
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
            f"Dartmouth base_url {url!r} is not a valid absolute URL. "
            f"Expected something like {DEFAULT_BASE_URL!r}."
        )
    if parts.scheme == "https":
        return cleaned
    if parts.scheme == "http" and (parts.hostname or "") in _PLAINTEXT_OK_HOSTS:
        return cleaned
    raise InsecureEndpoint(
        f"Dartmouth base_url {url!r} uses {parts.scheme!r}, which would send "
        f"the API key unencrypted to {parts.hostname!r}. Use https:// "
        f"(plaintext http:// is permitted only for "
        f"{', '.join(sorted(_PLAINTEXT_OK_HOSTS))})."
    )


class DartmouthModel(Model):
    """A model served by the Dartmouth Chat OpenAI-compatible gateway."""

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
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
            name: A Dartmouth model id, e.g. ``qwen.qwen3.5-122b``.
            api_key: Bearer token. Resolved from the environment or the local
                credential stores when omitted.
            base_url: Gateway root. Must be HTTPS unless it is loopback.
            cost: Real pricing from the live catalog, normally supplied by
                :meth:`DartmouthProvider.create_model`. Omitting it means the
                price is **unknown**, which is treated as paid -- see
                :meth:`_enforce_cost_policy`.

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
            provider="dartmouth",
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
            # model to free let `DartmouthModel(name="anthropic.claude-opus-5")`
            # skip the cost gate entirely and bill a real account.
            cost=cost if cost is not None else ModelCost(is_free=False),
            **kwargs,
        )

        credential = (
            None if api_key else resolve_dartmouth_api_key(required=True)
        )
        self._api_key = api_key or (credential.key if credential else "")
        self._base_url = validate_base_url(base_url or DEFAULT_BASE_URL)
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
                "Dartmouth credential %s resolved from %s",
                mask_key(self._api_key),
                credential.source,
            )

        self._enforce_cost_policy()

    def _enforce_cost_policy(self) -> None:
        """Refuse a paid model unless the operator opted in.

        Checked at construction rather than at call time so the failure lands
        where the model was chosen, not deep inside a pipeline run.

        Unknown pricing is refused alongside known-paid pricing. The catalog is
        the only authority on what a model costs, so a model built without
        consulting it has an unknown price -- and treating unknown as free is
        precisely the assumption that spends money by accident.
        """
        if paid_models_allowed():
            return
        if not self._pricing_is_known:
            raise PaidModelRefused(
                f"{self.name!r} was constructed without pricing, so its cost "
                f"is unknown and it is treated as paid. Build it through "
                f"DartmouthProvider.create_model(), which attaches real "
                f"pricing from the live catalog, or pass an explicit cost=. "
                f"Set {ALLOW_PAID_ENV_VAR}=1 to permit unpriced and paid usage."
            )
        if self.cost.is_free:
            return
        raise PaidModelRefused(
            f"{self.name!r} costs money "
            f"(input ${self.cost.input_cost_per_1k_tokens:.6f}/1k, "
            f"output ${self.cost.output_cost_per_1k_tokens:.6f}/1k) and "
            f"{ALLOW_PAID_ENV_VAR} is not set to '1'. Dartmouth Chat serves "
            f"free models -- see DartmouthProvider.list_free_models() -- or "
            f"set {ALLOW_PAID_ENV_VAR}=1 to permit paid usage."
        )

    @property
    def api_key_is_set(self) -> bool:
        """Whether a credential is available. Never exposes the key itself."""
        return bool(self._api_key)

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

    async def __aenter__(self) -> "DartmouthModel":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON to the gateway and return the decoded response.

        Transport failures (connection reset, DNS blip) are retried
        ``max_retries`` times. A *model* being down is deliberately NOT
        retried: the backend stays down for minutes, so retrying wastes the
        caller's time where :meth:`DartmouthProvider.generate_free` would
        simply move to the next free model.
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
                            f"Dartmouth Chat returned HTTP {response.status} "
                            f"for model {self.name!r}: {_safe_error_body(body)}"
                        )
                        # A downed model backend is reported as a 400/500 from
                        # the gateway, which is indistinguishable from a bad
                        # request unless the body is inspected. Callers need
                        # to tell them apart to fall back sensibly.
                        if response.status >= 500 or _looks_unavailable(body):
                            raise ModelUnavailable(detail)
                        raise DartmouthModelError(detail)
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError as exc:
                        # A maintenance window serves an HTML redirect page.
                        raise DartmouthModelError(
                            f"Dartmouth Chat returned a non-JSON response for "
                            f"{self.name!r} (is the gateway in maintenance?): "
                            f"{_safe_error_body(body)}"
                        ) from exc
            except aiohttp.ClientError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "Dartmouth transport error for %s (attempt %d/%d), "
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

        raise DartmouthModelError(
            f"Dartmouth Chat request failed for {self.name!r} after "
            f"{self._max_retries + 1} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _extract_text(response: Dict[str, Any], model_name: str) -> str:
        """Pull the assistant text out of a chat-completion response.

        Reasoning models put their scratchpad in ``reasoning_content`` and the
        answer in ``content``. When the token budget is exhausted while
        thinking, ``content`` is null -- which is a truncation, not an empty
        answer, and must not be returned as an empty string.
        """
        choices = response.get("choices") or []
        if not choices:
            raise DartmouthModelError(
                f"Dartmouth Chat returned no choices for {model_name!r}"
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
        raise DartmouthModelError(
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

        Extra ``kwargs`` are forwarded to the gateway as request fields, so
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
                "model": self.name,
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

        Not every model behind this gateway supports a response-format
        parameter, so the schema is stated in the prompt and the reply is
        parsed. A reply that is not valid JSON raises rather than being
        silently coerced into a string.
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
            raise DartmouthModelError(
                f"{self.name!r} did not return valid JSON: {exc}. "
                f"Response began: {text[:200]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise DartmouthModelError(
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
            raise DartmouthModelError(
                f"{self.name!r} returned JSON that does not match the "
                f"requested schema at {list(exc.absolute_path) or '<root>'}: "
                f"{exc.message}. Received: {json.dumps(parsed)[:200]}"
            ) from exc
        except jsonschema.SchemaError as exc:
            raise DartmouthModelError(
                f"the schema passed to generate_structured() is itself "
                f"invalid: {exc.message}"
            ) from exc
        return parsed

    async def health_check(self) -> bool:
        """Whether the gateway will serve this model."""
        try:
            await self.generate("ping", temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS)
            return True
        except Exception as exc:  # noqa: BLE001 - health checks report, not raise
            logger.warning("Dartmouth health check failed for %s: %s", self.name, exc)
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
        """Estimate USD cost. Exactly 0.0 for the free models.

        Raises:
            DartmouthModelError: if the model was built without pricing. The
                zero-filled default would otherwise report $0.00 for a model
                that may well bill -- a confidently wrong budget number is
                worse than an error.
        """
        if not self._pricing_is_known:
            raise DartmouthModelError(
                f"cannot estimate cost for {self.name!r}: it was constructed "
                f"without pricing. Build it through "
                f"DartmouthProvider.create_model() to attach real pricing."
            )
        if self.cost.is_free:
            return 0.0
        # ~4 characters per token is the usual rough English estimate; this is
        # a budgeting aid, not billing.
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max_tokens or DEFAULT_MAX_TOKENS
        return self.cost.calculate_cost(input_tokens, output_tokens)

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

import json
import logging
import os
from typing import Any, Dict, Optional

from ..core.model import Model, ModelCapabilities, ModelCost, ModelRequirements
from .dartmouth_credentials import mask_key, resolve_dartmouth_api_key

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOW_PAID_ENV_VAR",
    "DEFAULT_BASE_URL",
    "DartmouthModel",
    "DartmouthModelError",
    "PaidModelRefused",
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

_REQUEST_TIMEOUT_SECONDS = 300


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
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            name: A Dartmouth model id, e.g. ``qwen.qwen3.5-122b``.
            api_key: Bearer token. Resolved from the environment or the local
                credential stores when omitted.
            base_url: Gateway root. Defaults to Dartmouth Chat.
            cost: Pricing. Defaults to free; the provider supplies real
                pricing from the live catalog.
        """
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
            cost=cost or ModelCost(is_free=True),
            **kwargs,
        )

        credential = (
            None if api_key else resolve_dartmouth_api_key(required=True)
        )
        self._api_key = api_key or (credential.key if credential else "")
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._is_available = True

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
        """
        if self.cost.is_free or paid_models_allowed():
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

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON to the gateway and return the decoded response."""
        import aiohttp

        url = f"{self._base_url}/{path.lstrip('/')}"
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                ) as response:
                    body = await response.text()
                    if response.status >= 400:
                        # The body may echo the request but never the bearer
                        # token, so it is safe to surface.
                        detail = (
                            f"Dartmouth Chat returned HTTP {response.status} "
                            f"for model {self.name!r}: {body[:500]}"
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
                            f"{body[:200]}"
                        ) from exc
        except aiohttp.ClientError as exc:
            raise DartmouthModelError(
                f"Dartmouth Chat request failed for {self.name!r}: {exc}"
            ) from exc

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
        """Generate text from ``prompt``."""
        messages = []
        system_prompt = kwargs.pop("system_prompt", None)
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        }
        payload.update(kwargs)

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
        return parsed

    async def health_check(self) -> bool:
        """Whether the gateway will serve this model."""
        try:
            await self.generate("ping", temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS)
            return True
        except Exception as exc:  # noqa: BLE001 - health checks report, not raise
            logger.warning("Dartmouth health check failed for %s: %s", self.name, exc)
            return False

    async def estimate_cost(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> float:
        """Estimate USD cost. Exactly 0.0 for the free models."""
        if self.cost.is_free:
            return 0.0
        # ~4 characters per token is the usual rough English estimate; this is
        # a budgeting aid, not billing.
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max_tokens or DEFAULT_MAX_TOKENS
        return self.cost.calculate_cost(input_tokens, output_tokens)

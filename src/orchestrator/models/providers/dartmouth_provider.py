"""Dartmouth Chat provider.

Model ids and pricing come from the live catalog at
``https://chat.dartmouth.edu/api/models``. Nothing is hard-coded: this project
has already shipped two rounds of hard-coded Anthropic model ids that rotted
(retired 2024 dates, then invented ``-latest`` aliases), and a catalog whose
free/paid split determines spending is exactly the wrong thing to guess at.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ...core.model import ModelCapabilities, ModelCost, ModelRequirements
from ..dartmouth_credentials import resolve_dartmouth_api_key
from ..dartmouth_model import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DartmouthModel,
    DartmouthModelError,
    ModelUnavailable,
    ReasoningTruncated,
    validate_base_url,
)
from .base import ModelProvider, ProviderConfig

logger = logging.getLogger(__name__)

__all__ = ["DartmouthProvider"]


def _walk_token_costs(node: object) -> Tuple[List[float], List[float]]:
    """Collect every input/output cost-per-token found anywhere in ``node``.

    The catalog nests pricing at different depths: internal models under
    ``upstream_model_info.model_info``, external ones a level deeper. A
    recursive search is more robust than tracking those shapes.
    """
    inputs: List[float] = []
    outputs: List[float] = []

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            value = obj.get("input_cost_per_token")
            if value is not None:
                inputs.append(float(value))
            value = obj.get("output_cost_per_token")
            if value is not None:
                outputs.append(float(value))
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(node)
    return inputs, outputs


def fetch_catalog_sync(
    base_url: str, api_key: str, timeout: float
) -> Dict[str, Dict[str, Any]]:
    """Fetch the model catalog synchronously, keyed by model id.

    The async path is preferred everywhere else. This exists for
    :func:`orchestrator.populate_model_registry`, which is synchronous and can
    be reached from inside a running event loop -- where ``asyncio.run`` would
    raise. Uses ``urllib`` rather than aiohttp for exactly that reason.
    """
    import json
    import urllib.request

    url = f"{validate_base_url(base_url)}/models"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {e["id"]: e for e in (payload.get("data") or []) if e.get("id")}


def free_models_from_catalog(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, ModelCost]:
    """The subset of ``catalog`` that costs nothing per token."""
    priced = {mid: model_cost_from_catalog(e) for mid, e in catalog.items()}
    return {mid: cost for mid, cost in priced.items() if cost.is_free}


def model_cost_from_catalog(entry: Dict[str, Any]) -> ModelCost:
    """Build a :class:`ModelCost` from one catalog entry.

    An entry with **no** pricing at all (embeddings, helper bots) is treated
    as paid, not free. Absence of a price is not evidence of zero price, and
    guessing "free" here is the failure mode that spends real money.
    """
    inputs, outputs = _walk_token_costs(entry)
    if not inputs and not outputs:
        return ModelCost(is_free=False)

    max_input = max(inputs) if inputs else 0.0
    max_output = max(outputs) if outputs else 0.0
    return ModelCost(
        input_cost_per_1k_tokens=max_input * 1000,
        output_cost_per_1k_tokens=max_output * 1000,
        is_free=(max_input == 0.0 and max_output == 0.0),
    )


class DartmouthProvider(ModelProvider):
    """Serves models from the Dartmouth Chat OpenAI-compatible gateway."""

    def __init__(self, config: Optional[ProviderConfig] = None) -> None:
        # ProviderConfig defaults to a 30s timeout, which is right for a
        # metadata call and far too short for generation on a busy shared
        # cluster. The default config therefore carries the generation
        # timeout; an explicit config is honoured exactly as given.
        super().__init__(
            config
            or ProviderConfig(
                name="dartmouth", timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS
            )
        )
        # The catalog request carries the bearer token too, so the endpoint is
        # checked here as well -- not only in DartmouthModel.
        self._base_url = validate_base_url(self.config.base_url or DEFAULT_BASE_URL)
        self._catalog: Dict[str, Dict[str, Any]] = {}
        self._costs: Dict[str, ModelCost] = {}

    async def initialize(self) -> None:
        """Resolve credentials and load the model catalog."""
        if not self.config.api_key:
            self.config.api_key = resolve_dartmouth_api_key(required=True).key
        await self._load_catalog()
        self._initialized = True

    async def _load_catalog(self) -> None:
        """Fetch and index the live model catalog."""
        import aiohttp

        url = f"{self._base_url}/models"
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url, headers={"Authorization": f"Bearer {self.config.api_key}"}
                ) as response:
                    if response.status >= 400:
                        raise DartmouthModelError(
                            f"Dartmouth model catalog returned HTTP "
                            f"{response.status}: {(await response.text())[:300]}"
                        )
                    payload = await response.json()
        except aiohttp.ClientError as exc:
            raise DartmouthModelError(
                f"Could not reach the Dartmouth model catalog: {exc}"
            ) from exc

        entries = payload.get("data") or []
        self._catalog = {e["id"]: e for e in entries if e.get("id")}
        self._costs = {
            model_id: model_cost_from_catalog(entry)
            for model_id, entry in self._catalog.items()
        }
        free = sorted(m for m, c in self._costs.items() if c.is_free)
        logger.info(
            "Dartmouth catalog: %d models, %d free (%s)",
            len(self._catalog),
            len(free),
            ", ".join(free) or "none",
        )

    def _require_catalog(self) -> None:
        if not self._catalog:
            raise DartmouthModelError(
                "Dartmouth catalog not loaded; call await provider.initialize()"
            )

    def list_free_models(self) -> List[str]:
        """Model ids that cost nothing per token."""
        self._require_catalog()
        return sorted(m for m, cost in self._costs.items() if cost.is_free)

    def list_paid_models(self) -> List[str]:
        """Model ids that cost money, or whose price is unknown."""
        self._require_catalog()
        return sorted(m for m, cost in self._costs.items() if not cost.is_free)

    async def create_model(self, model_name: str, **kwargs: Any) -> DartmouthModel:
        """Build a model, carrying its real catalog pricing.

        Pricing is attached here so :class:`DartmouthModel` can refuse a paid
        model without a second network round trip.
        """
        self._require_catalog()
        if model_name not in self._catalog:
            available = ", ".join(sorted(self._catalog)[:10])
            raise DartmouthModelError(
                f"{model_name!r} is not served by Dartmouth Chat. "
                f"Available include: {available}..."
            )
        return DartmouthModel(
            name=model_name,
            api_key=self.config.api_key,
            base_url=self._base_url,
            cost=self._costs[model_name],
            capabilities=self.get_model_capabilities(model_name),
            requirements=self.get_model_requirements(model_name),
            # The provider's configured transport policy applies to every
            # model it builds; previously these were hard-coded and the
            # config was silently ignored.
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            **kwargs,
        )

    #: Preference order for free models when the caller does not name one.
    #: Larger, stronger models first; the small Llama is the reliable floor.
    #: Matched as substrings so a version bump in the catalog does not strand
    #: this list -- anything unmatched still gets used, just later.
    _FREE_PREFERENCE = (
        "qwen3.5",       # largest general reasoning model in the free set
        "gpt-oss-120b",
        "gemma-4",
        "gemma-3",
        "qwen3-vl",
        "llama-3.2-11b",
        "llama-3-2-3b",  # smallest and fastest; dependable fallback
    )

    def free_models_by_preference(self) -> List[str]:
        """Free models ordered strongest-first, unknown ones last."""
        free = self.list_free_models()

        def rank(model_id: str) -> Tuple[int, str]:
            lowered = model_id.lower()
            for index, marker in enumerate(self._FREE_PREFERENCE):
                if marker in lowered:
                    return (index, model_id)
            return (len(self._FREE_PREFERENCE), model_id)

        return sorted(free, key=rank)

    async def generate_free(
        self,
        prompt: str,
        *,
        models: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Generate using the first free model whose backend is up.

        Each free model is served from its own cluster endpoint, and those go
        down independently -- a real observed failure is::

            Cannot connect to host vllm-qwen35.ai-prod.svc.cluster.local:8000

        A single-model call strands the caller whenever that happens, so this
        walks the preference order and only gives up when every candidate is
        unavailable.

        Returns:
            ``(text, model_id)`` -- the reply and which model produced it, so
            callers can record what actually answered.

        Raises:
            DartmouthModelError: if no free model could serve the request. A
            non-availability error (a genuinely bad request) is raised
            immediately rather than retried against every model.
        """
        candidates = models or self.free_models_by_preference()
        if not candidates:
            raise DartmouthModelError("no free Dartmouth models are available")

        skipped: List[str] = []
        for model_id in candidates:
            model = await self.create_model(model_id)
            try:
                return await model.generate(prompt, **kwargs), model_id
            except ModelUnavailable as exc:
                logger.warning("Free model %s is down, trying next: %s", model_id, exc)
                skipped.append(f"{model_id} (backend down)")
                continue
            except ReasoningTruncated as exc:
                # Several free models are reasoning models that spend the
                # whole budget thinking. Rather than fail, hand the request to
                # the next candidate -- a non-reasoning model answers the same
                # prompt comfortably within the same budget.
                logger.warning(
                    "Free model %s exhausted its budget reasoning, trying "
                    "next: %s",
                    model_id,
                    exc,
                )
                skipped.append(f"{model_id} (reasoning truncated)")
                continue
            finally:
                # Each candidate holds its own HTTP session. Walking a chain
                # of downed models would otherwise leak one session per
                # attempt. The success path closes too: the reply is already
                # in hand by the time this runs.
                await model.aclose()

        raise DartmouthModelError(
            f"no free Dartmouth model could answer: {', '.join(skipped)}. "
            f"If every entry says 'reasoning truncated', raise max_tokens "
            f"(current default {DEFAULT_MAX_TOKENS})."
        )

    async def health_check(self) -> bool:
        """Whether the catalog is reachable with the configured credential."""
        try:
            await self._load_catalog()
            return bool(self._catalog)
        except Exception as exc:  # noqa: BLE001 - health checks report
            logger.warning("Dartmouth health check failed: %s", exc)
            return False

    async def discover_models(self) -> List[str]:
        """Every model id the gateway serves."""
        if not self._catalog:
            await self._load_catalog()
        return sorted(self._catalog)

    def get_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """Capabilities inferred from the catalog entry."""
        entry = self._catalog.get(model_name, {})
        tasks = ["generate", "analyze", "transform", "summarize"]
        # Vision models advertise it in their id; the catalog has no uniform
        # modality field across internal and external models.
        vision = "vision" in model_name.lower() or "-vl" in model_name.lower()
        if vision:
            tasks.append("vision")
        context_window = 0
        for key in ("max_model_len", "context_length", "max_context_length"):
            value = entry.get(key)
            if isinstance(value, int) and value > 0:
                context_window = value
                break
        return ModelCapabilities(
            supported_tasks=tasks,
            context_window=context_window or 32768,
            supports_structured_output=True,
            vision_capable=vision,
        )

    def get_model_requirements(self, model_name: str) -> ModelRequirements:
        """Local requirements for a remotely hosted model.

        Inference runs on Dartmouth's cluster, so the local cost is just an
        HTTP request. ``ModelRequirements`` forbids zero, so the minimum
        defaults stand in for "negligible" -- overstating slightly is safer
        than a field that cannot be constructed.
        """
        return ModelRequirements(requires_gpu=False)

    def get_model_cost(self, model_name: str) -> ModelCost:
        """Live pricing for ``model_name``."""
        self._require_catalog()
        if model_name not in self._costs:
            raise DartmouthModelError(f"{model_name!r} is not in the catalog")
        return self._costs[model_name]

"""HuggingFace Inference API provider.

Model ids, pricing and route availability come from the live catalog at
``https://router.huggingface.co/v1/models``. Nothing is hard-coded: which
models carry a free route changes upstream (promos start and end), and a
stale local list would either miss free models or -- much worse -- offer a
paid one as if it were free.

Cost semantics, verified against the live catalog (2026-08-21): each catalog
entry carries a ``providers`` list whose entries report ``status``
(``live``/``error``), ``pricing`` in USD **per million** tokens when
available, and an ``is_free`` promo flag when applicable. A model is free
only while a **live** provider marks it free or prices it at exactly zero --
and because an unpinned request routes ``:fastest``, a free model is pinned
to its free provider on the wire (``model_id:provider``) so the router cannot
send it to a paid one.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ...core.model import ModelCapabilities, ModelCost, ModelRequirements
from ..huggingface_credentials import resolve_huggingface_api_key
from ..huggingface_model import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    HuggingFaceInferenceModel,
    HuggingFaceModelError,
    ModelUnavailable,
    ReasoningTruncated,
    validate_base_url,
)
from .base import ModelProvider, ProviderConfig

logger = logging.getLogger(__name__)

__all__ = [
    "HuggingFaceProvider",
    "catalog_price_is_known",
    "fetch_catalog_sync",
    "free_models_from_catalog",
    "free_route_from_catalog",
    "model_cost_from_catalog",
]


def _live_providers(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The provider routes reported as serving the model right now.

    A route in ``error`` state is not a route: counting it would let a downed
    free route mark a model free, or a downed paid route inflate its budget
    estimate.
    """
    return [
        p
        for p in (entry.get("providers") or [])
        if isinstance(p, dict) and p.get("status") == "live"
    ]


def _explicit_pricing(provider: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """``(input, output)`` USD per million tokens, or None when unreported."""
    pricing = provider.get("pricing")
    if not isinstance(pricing, dict):
        return None
    input_price = pricing.get("input")
    output_price = pricing.get("output")
    if input_price is None or output_price is None:
        return None
    return float(input_price), float(output_price)


def _is_free_route(provider: Dict[str, Any]) -> bool:
    """Whether this live route currently costs nothing.

    The ``is_free`` promo flag wins when present; otherwise explicit zero
    pricing counts. Absent pricing does NOT -- absence of a price is not
    evidence of zero price, and guessing "free" here is the failure mode that
    spends real money.
    """
    if provider.get("is_free") is True:
        return True
    pricing = _explicit_pricing(provider)
    return pricing is not None and pricing == (0.0, 0.0)


def free_route_from_catalog(entry: Dict[str, Any]) -> Optional[str]:
    """The provider a free model must be pinned to, or None if it is not free.

    The pin matters: an unpinned request routes ``:fastest``, and the fastest
    provider is not necessarily the free one.
    """
    for provider in _live_providers(entry):
        if _is_free_route(provider) and provider.get("provider"):
            return str(provider["provider"])
    return None


def model_cost_from_catalog(entry: Dict[str, Any]) -> ModelCost:
    """Build a :class:`ModelCost` from one catalog entry.

    Free when a live provider offers a free route. Otherwise the most
    expensive live route sets the estimate: routing is server-side, so
    budgeting on the cheapest route would understate. An entry with **no**
    pricing at all is treated as paid, not free.
    """
    live = _live_providers(entry)
    if any(_is_free_route(p) for p in live):
        return ModelCost(is_free=True)

    priced = [p for p in (_explicit_pricing(p) for p in live) if p is not None]
    if not priced:
        return ModelCost(is_free=False)

    # The router reports USD per million tokens; ModelCost is per 1k.
    max_input = max(input_price for input_price, _ in priced) / 1000
    max_output = max(output_price for _, output_price in priced) / 1000
    return ModelCost(
        input_cost_per_1k_tokens=max_input,
        output_cost_per_1k_tokens=max_output,
        is_free=False,
    )


def free_models_from_catalog(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, ModelCost]:
    """The subset of ``catalog`` that currently has a free route."""
    priced = {mid: model_cost_from_catalog(e) for mid, e in catalog.items()}
    return {mid: cost for mid, cost in priced.items() if cost.is_free}


def catalog_price_is_known(entry: Dict[str, Any]) -> bool:
    """Whether the catalog says anything definite about this model's price.

    A model with a free route is known-free; a model with explicit pricing on
    a live route is known-paid. Anything else is *unknown* -- and unknown must
    behave as paid without pretending a $0.00 estimate is real.
    """
    live = _live_providers(entry)
    return any(
        _is_free_route(p) or _explicit_pricing(p) is not None for p in live
    )


def _probed_throughput(entry: Dict[str, Any]) -> Optional[float]:
    """The fastest probed throughput among live routes, when reported."""
    throughputs = [
        float(p["throughput"])
        for p in _live_providers(entry)
        if isinstance(p.get("throughput"), (int, float))
    ]
    return max(throughputs) if throughputs else None


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


class HuggingFaceProvider(ModelProvider):
    """Serves chat models from the HuggingFace Inference Providers router."""

    def __init__(self, config: Optional[ProviderConfig] = None) -> None:
        # ProviderConfig defaults to a 30s timeout, which is right for a
        # metadata call and far too short for generation behind a cold-starting
        # model. The default config therefore carries the generation timeout;
        # an explicit config is honoured exactly as given.
        super().__init__(
            config
            or ProviderConfig(
                name="huggingface", timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS
            )
        )
        # The catalog request carries the bearer token too, so the endpoint is
        # checked here as well -- not only in HuggingFaceInferenceModel.
        self._base_url = validate_base_url(self.config.base_url or DEFAULT_BASE_URL)
        self._catalog: Dict[str, Dict[str, Any]] = {}
        self._costs: Dict[str, ModelCost] = {}

    async def initialize(self) -> None:
        """Resolve credentials and load the model catalog."""
        if not self.config.api_key:
            self.config.api_key = resolve_huggingface_api_key(required=True).key
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
                        raise HuggingFaceModelError(
                            f"HuggingFace model catalog returned HTTP "
                            f"{response.status}: {(await response.text())[:300]}"
                        )
                    payload = await response.json()
        except aiohttp.ClientError as exc:
            raise HuggingFaceModelError(
                f"Could not reach the HuggingFace model catalog: {exc}"
            ) from exc

        entries = payload.get("data") or []
        self._catalog = {e["id"]: e for e in entries if e.get("id")}
        self._costs = {
            model_id: model_cost_from_catalog(entry)
            for model_id, entry in self._catalog.items()
        }
        free = sorted(m for m, c in self._costs.items() if c.is_free)
        logger.info(
            "HuggingFace catalog: %d models, %d with a free route (%s)",
            len(self._catalog),
            len(free),
            ", ".join(free) or "none",
        )

    def _require_catalog(self) -> None:
        if not self._catalog:
            raise HuggingFaceModelError(
                "HuggingFace catalog not loaded; call await provider.initialize()"
            )

    def list_free_models(self) -> List[str]:
        """Model ids that currently have a zero-cost route."""
        self._require_catalog()
        return sorted(m for m, cost in self._costs.items() if cost.is_free)

    def list_paid_models(self) -> List[str]:
        """Model ids that cost money, or whose price is unknown."""
        self._require_catalog()
        return sorted(m for m, cost in self._costs.items() if not cost.is_free)

    async def create_model(
        self, model_name: str, **kwargs: Any
    ) -> HuggingFaceInferenceModel:
        """Build a model, carrying its real catalog pricing and free-route pin.

        Pricing is attached here so :class:`HuggingFaceInferenceModel` can
        refuse a paid model without a second network round trip.
        """
        self._require_catalog()
        if model_name not in self._catalog:
            available = ", ".join(sorted(self._catalog)[:10])
            raise HuggingFaceModelError(
                f"{model_name!r} is not served by the HuggingFace router. "
                f"Available include: {available}..."
            )
        entry = self._catalog[model_name]
        # An unpriced entry gets cost=None, not a zero-filled ModelCost: the
        # model then treats its price as *unknown* -- refused without the
        # opt-in, and estimate_cost raises rather than reporting a
        # confidently wrong $0.00 for a model that may bill.
        cost = self._costs[model_name] if catalog_price_is_known(entry) else None
        return HuggingFaceInferenceModel(
            name=model_name,
            api_key=self.config.api_key,
            base_url=self._base_url,
            route=free_route_from_catalog(entry),
            cost=cost,
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

    def free_models_by_preference(self) -> List[str]:
        """Free models ordered by probed throughput, unprobed ones last.

        Unlike Dartmouth there is no stable, known free set to hard-code a
        preference list against -- promos start and end upstream. The
        catalog's own probe data is the honest ordering signal.
        """
        free = self.list_free_models()

        def rank(model_id: str) -> Tuple[int, float, str]:
            throughput = _probed_throughput(self._catalog.get(model_id, {}))
            if throughput is None:
                return (1, 0.0, model_id)
            return (0, -throughput, model_id)

        return sorted(free, key=rank)

    async def generate_free(
        self,
        prompt: str,
        *,
        models: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Generate using the first free-routed model whose backend answers.

        Free routes are promos on individual providers and they flap. A
        single-model call strands the caller whenever that happens, so this
        walks the preference order and only gives up when every candidate is
        unavailable. A rate limit is account-level, so it is NOT walked past:
        every other attempt would draw on the same quota.

        Returns:
            ``(text, model_id)`` -- the reply and which model produced it, so
            callers can record what actually answered.

        Raises:
            HuggingFaceModelError: if no free model could serve the request.
            RateLimited: if the account is throttled (propagates rather than
                burning through the candidate list).
        """
        candidates = models or self.free_models_by_preference()
        if not candidates:
            raise HuggingFaceModelError(
                "no HuggingFace models with a free route are available"
            )

        skipped: List[str] = []
        for model_id in candidates:
            model = await self.create_model(model_id)
            try:
                return await model.generate(prompt, **kwargs), model_id
            except ModelUnavailable as exc:
                # Includes ModelLoading: a cold-starting model is handed to
                # the next candidate rather than waited on.
                logger.warning(
                    "Free model %s is not serving, trying next: %s", model_id, exc
                )
                skipped.append(f"{model_id} (unavailable)")
                continue
            except ReasoningTruncated as exc:
                # Several candidates are reasoning models that spend the whole
                # budget thinking. Rather than fail, hand the request to the
                # next candidate -- a non-reasoning model answers the same
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

        raise HuggingFaceModelError(
            f"no free HuggingFace model could answer: {', '.join(skipped)}. "
            f"If every entry says 'reasoning truncated', raise max_tokens "
            f"(current default {DEFAULT_MAX_TOKENS})."
        )

    async def health_check(self) -> bool:
        """Whether the catalog is reachable with the configured credential."""
        try:
            await self._load_catalog()
            return bool(self._catalog)
        except Exception as exc:  # noqa: BLE001 - health checks report
            logger.warning("HuggingFace health check failed: %s", exc)
            return False

    async def discover_models(self) -> List[str]:
        """Every chat model id the router serves."""
        if not self._catalog:
            await self._load_catalog()
        return sorted(self._catalog)

    def get_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """Capabilities inferred from the catalog entry."""
        entry = self._catalog.get(model_name, {})
        tasks = ["generate", "analyze", "transform", "summarize"]
        architecture = entry.get("architecture") or {}
        modalities = architecture.get("input_modalities") or []
        vision = "image" in modalities
        if vision:
            tasks.append("vision")
        live = _live_providers(entry)
        context_lengths = [
            int(p["context_length"])
            for p in live
            if isinstance(p.get("context_length"), (int, float))
            and p["context_length"] > 0
        ]
        return ModelCapabilities(
            supported_tasks=tasks,
            context_window=max(context_lengths) if context_lengths else 32768,
            supports_structured_output=any(
                p.get("supports_structured_output") for p in live
            ),
            supports_tools=any(p.get("supports_tools") for p in live),
            vision_capable=vision,
        )

    def get_model_requirements(self, model_name: str) -> ModelRequirements:
        """Local requirements for a remotely hosted model.

        Inference runs on the provider's infrastructure, so the local cost is
        just an HTTP request. ``ModelRequirements`` forbids zero, so the
        minimum defaults stand in for "negligible" -- overstating slightly is
        safer than a field that cannot be constructed.
        """
        return ModelRequirements(requires_gpu=False)

    def get_model_cost(self, model_name: str) -> ModelCost:
        """Live pricing for ``model_name``."""
        self._require_catalog()
        if model_name not in self._costs:
            raise HuggingFaceModelError(f"{model_name!r} is not in the catalog")
        return self._costs[model_name]

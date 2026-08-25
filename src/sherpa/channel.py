"""Model-boundary channels for sherpa (issue #492 provider-independence answer).

The model call is the one external boundary the MVP treats specially:
``RecordedChannel`` replays recorded responses deterministically inside the
hermetic acceptance suite, ``LiveChannel`` lazily uses orchestrator's supported
providers when credentials exist, and ``EchoChannel`` refuses use so a
deterministic run fails loudly if it unexpectedly needs a model.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

from pydantic import BaseModel


class ChannelResponse(BaseModel):
    """One model reply plus what it cost.

    ``tokens_estimated`` distinguishes "the provider reported this" from "we
    estimated it because the provider did not". Reporting an unmeasured 0 as if
    it were measured made ``Budgets.max_tokens`` and ``max_cost_usd``
    unenforceable against real model spend.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    tokens_estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(text: str) -> int:
    """Deterministic, provider-independent token estimate.

    Roughly 4 characters per token, floored at 1 for non-empty text. Used only
    when a provider does not report usage; the result is always flagged with
    ``tokens_estimated=True`` so a budget is enforced conservatively rather
    than not at all.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class ModelChannel(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        session: str,
    ) -> ChannelResponse: ...


class ChannelRequired(Exception):
    """Raised when a run needs a model but none is configured."""


class RecordingExhausted(Exception):
    """Raised when a session role asks for more responses than were recorded."""


class ProviderUnavailable(Exception):
    """Raised when no supported live provider (Dartmouth/HF) is reachable or keyed."""


class RecordedChannel:
    """Deterministic FIFO replay of recorded responses, keyed by session role."""

    def __init__(self, recordings: dict[str, list[str]]) -> None:
        self.recordings = {role: list(resps) for role, resps in recordings.items()}

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        session: str,
    ) -> ChannelResponse:
        queue = self.recordings.get(session)
        if not queue:
            raise RecordingExhausted(f"no recorded response left for session {session!r}")
        text = queue.pop(0)
        return ChannelResponse(
            text=text,
            model="recorded",
            prompt_tokens=sum(len(str(m)) for m in messages) // 4,
            completion_tokens=len(text) // 4,
        )


class EchoChannel:
    """Refuses every call: deterministic runs must not need a model."""

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        session: str,
    ) -> ChannelResponse:
        raise ChannelRequired(
            f"session {session!r} requested a model completion but this run "
            "has no channel configured; supply recordings or policy='live'"
        )


def _import_dartmouth() -> Any:
    mod = importlib.import_module("orchestrator")
    return getattr(mod, "DartmouthProvider", None)


def _import_hf_provider() -> Any:
    try:
        mod = importlib.import_module("orchestrator.models.providers.huggingface")
    except ImportError:
        return None
    return getattr(mod, "HuggingFaceProvider", None)


class LiveChannel:
    """Best-effort adapter over orchestrator's two supported providers.

    Dartmouth Chat free models are tried first (``generate_free``), then the
    HuggingFace Inference API provider. Retired providers are never contacted.
    """

    def __init__(self, prefer_free: bool = True) -> None:
        self.prefer_free = prefer_free
        self.model_used: str | None = None

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        session: str,
    ) -> ChannelResponse:
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        if self.prefer_free:
            dartmouth = _import_dartmouth()
            if dartmouth is not None:
                import asyncio

                async def _call() -> tuple[str, str]:
                    provider = dartmouth()
                    await provider.initialize()
                    return await provider.generate_free(prompt)

                try:
                    text, model_used = asyncio.run(_call())
                    self.model_used = model_used
                    return ChannelResponse(
                        text=text, model=str(model_used),
                        prompt_tokens=estimate_tokens(prompt),
                        completion_tokens=estimate_tokens(text),
                        tokens_estimated=True,
                    )
                except Exception as exc:  # noqa: BLE001 - fall through to next provider
                    # Remember why the preferred provider was skipped; this used
                    # to be swallowed entirely, so the real cause never surfaced.
                    self.last_error = f"dartmouth unavailable: {type(exc).__name__}: {exc}"
        hf_cls = _import_hf_provider()
        if hf_cls is not None:
            try:
                provider = hf_cls()
                text = provider.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)  # type: ignore[attr-defined]
                self.model_used = "huggingface"
                return ChannelResponse(
                    text=text, model="huggingface",
                    prompt_tokens=estimate_tokens(prompt),
                    completion_tokens=estimate_tokens(text),
                    tokens_estimated=True,
                )
            except Exception as exc:  # noqa: BLE001 - boundary: report unavailability
                prior = f" (after {self.last_error})" if getattr(self, "last_error", None) else ""
                raise ProviderUnavailable(f"live providers failed: {exc}{prior}") from exc
        raise ProviderUnavailable(
            "no supported provider available: set DARTMOUTH_CHAT_API_KEY or HF_TOKEN"
        )


def make_channel(policy: str, recordings: dict[str, list[str]] | None = None) -> ModelChannel:
    """Factory: ``recorded`` (default echo without recordings) or ``live``."""
    if policy == "recorded":
        return RecordedChannel(recordings) if recordings else EchoChannel()
    if policy == "live":
        return LiveChannel()
    raise ValueError(f"unknown channel policy {policy!r}")

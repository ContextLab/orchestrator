"""Model-boundary channels for sherpa (issue #492 provider-independence answer).

The model call is the one external boundary the MVP treats specially:
``RecordedChannel`` replays recorded responses deterministically inside the
hermetic acceptance suite, ``LiveChannel`` lazily uses orchestrator's supported
providers when credentials exist, and ``EchoChannel`` refuses use so a
deterministic run fails loudly if it unexpectedly needs a model.
"""

from __future__ import annotations

import importlib
from typing import Any, NamedTuple, Protocol

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


class RecordingMismatch(Exception):
    """A request did not match the recorded request it would have been answered with."""


class ProviderUnavailable(Exception):
    """Raised when no supported live provider (Dartmouth/HF) is reachable or keyed."""


#: Request fields a recorded turn may be keyed on. ``session`` is not among
#: them: it selects the tape, it does not identify the request within it.
MATCHABLE_REQUEST_FIELDS: tuple[str, ...] = ("messages", "temperature", "max_tokens")


class RecordedTurn(NamedTuple):
    """One recorded response, optionally bound to the request that produced it."""

    text: str
    match: dict[str, Any] | None


def _coerce_turn(session: str, position: int, entry: Any) -> RecordedTurn:
    if isinstance(entry, str):
        return RecordedTurn(text=entry, match=None)
    if isinstance(entry, RecordedTurn):
        return entry
    if not isinstance(entry, dict):
        raise ValueError(
            f"recording {session!r}[{position}] must be a str or a dict with a "
            f"'text' key, got {type(entry).__name__}"
        )
    if "text" not in entry:
        raise ValueError(f"recording {session!r}[{position}] has no 'text' key: {entry!r}")
    match = entry.get("match")
    if match is None:
        return RecordedTurn(text=str(entry["text"]), match=None)
    if not isinstance(match, dict):
        raise ValueError(f"recording {session!r}[{position}] 'match' must be a dict, got {match!r}")
    unknown = sorted(set(match) - set(MATCHABLE_REQUEST_FIELDS))
    if unknown:
        raise ValueError(
            f"recording {session!r}[{position}] matches on unknown request "
            f"field(s) {unknown}; matchable fields are {list(MATCHABLE_REQUEST_FIELDS)}"
        )
    return RecordedTurn(text=str(entry["text"]), match=dict(match))


class RecordedChannel:
    """Deterministic replay of recorded responses, per session role.

    Each session holds an ordered tape. A tape entry is either:

    * ``"answer text"`` -- an **unkeyed** entry. It replays positionally and
      cannot detect a mis-ordered request. Kept because many callers (and
      ``sherpa.demos``) write ``RecordedChannel({"planner": ["a", "b"]})``.
    * ``{"text": ..., "match": {...}}`` -- a **request-keyed** entry. Every
      field named in ``match`` (see :data:`MATCHABLE_REQUEST_FIELDS`) must equal
      the incoming request, or :class:`RecordingMismatch` is raised naming both
      sides. Keying on ``session`` alone made this class a positional FIFO, so
      an unrecorded prompt silently received the next queued answer and any plan
      reordering produced wrong-but-plausible responses with no error.
    """

    def __init__(self, recordings: dict[str, list[Any]]) -> None:
        self.recordings: dict[str, list[RecordedTurn]] = {
            role: [_coerce_turn(role, i, entry) for i, entry in enumerate(resps)]
            for role, resps in recordings.items()
        }
        self._consumed: dict[str, int] = {}

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
        position = self._consumed.get(session, 0)
        turn = queue[0]
        if turn.match is not None:
            actual = {
                "messages": [dict(m) for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            for field in MATCHABLE_REQUEST_FIELDS:
                if field in turn.match and actual[field] != turn.match[field]:
                    raise RecordingMismatch(
                        f"session {session!r} recording #{position} was recorded for a "
                        f"different request: {field} mismatch\n"
                        f"  recorded: {turn.match[field]!r}\n"
                        f"  actual:   {actual[field]!r}"
                    )
        queue.pop(0)
        self._consumed[session] = position + 1
        return ChannelResponse(
            text=turn.text,
            model="recorded",
            prompt_tokens=sum(len(str(m)) for m in messages) // 4,
            completion_tokens=len(turn.text) // 4,
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


def make_channel(policy: str, recordings: dict[str, list[Any]] | None = None) -> ModelChannel:
    """Factory: ``recorded`` (echo when no recordings are configured) or ``live``.

    ``recordings=None`` means "no recordings configured" -> ``EchoChannel``,
    which refuses. ``recordings={}`` is a *deliberately empty* recording set and
    yields a ``RecordedChannel`` that raises ``RecordingExhausted``; collapsing
    the two made an empty tape indistinguishable from an unconfigured run.
    """
    if policy == "recorded":
        return EchoChannel() if recordings is None else RecordedChannel(recordings)
    if policy == "live":
        return LiveChannel()
    raise ValueError(f"unknown channel policy {policy!r}")

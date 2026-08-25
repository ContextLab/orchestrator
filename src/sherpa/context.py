"""Context and coordination substrate (#492 §5).

The guarantee is **bounded overview + lossless source addressability +
on-demand retrieval** — not lossless compression into a context window.
Operational journal entries are concise (`intent`/`decision`/`observation`/
`assumption`/`blocker`/`result`), never required private chain-of-thought.
Documents become immutable structural chunks (exact char spans, content
hashes) in SQLite FTS5; summaries form a DAG where every summary points to
ALL children with exact spans/hashes. Reads are lock-free scoped snapshots;
compare-and-swap protects only state transitions (see `sherpa.store`).

Every span declares the coordinate frame its offsets belong to, so a
consumer can never mis-resolve one frame for the other (see `SPAN_FRAMES`
and `build_summary`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sherpa.channel import ModelChannel
    from sherpa.store import Store

JOURNAL_KINDS = ("intent", "decision", "observation", "assumption", "blocker", "result")


class JournalError(ValueError):
    """Invalid journal entry kind."""


def journal(
    store: "Store",
    run_id: str,
    node_key: str | None,
    kind: str,
    text: str,
    refs: list[str] | None = None,
) -> None:
    """Append one operational journal event. Appends are atomic; no mutex."""
    if kind not in JOURNAL_KINDS:
        raise JournalError(f"invalid journal kind {kind!r}; expected one of {JOURNAL_KINDS}")
    store.append(
        _event(kind="journal_appended", run_id=run_id, node_key=node_key,
               payload={"kind": kind, "text": text, "refs": refs or []})
    )


def _event(**kw):  # local indirection keeps module import graph flat in tests
    from sherpa.events import Event

    return Event(**kw)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    start: int
    end: int
    sha: str


def chunk_document(doc_id: str, text: str, strategy: str = "paragraph") -> list[Chunk]:
    """Deterministic structural chunking with EXACT char offsets.

    ``paragraph`` splits on blank lines (tiny paragraphs merge forward);
    ``heading`` splits markdown on ATX headings, heading included in its span.
    Invariant: for every chunk, text[start:end] == chunk text.
    """
    from sherpa.store import content_hash

    if strategy not in ("paragraph", "heading"):
        raise ValueError(f"unknown chunking strategy {strategy!r}")

    segments: list[tuple[int, int]] = []
    if strategy == "paragraph":
        blocks = text.split("\n\n")
        cursor = 0
        for block in blocks:
            offset = text.find(block, cursor)
            if block.strip():
                segments.append((offset, offset + len(block.rstrip())))
            cursor = offset + len(block)
    else:  # heading
        import re

        matches = list(re.finditer(r"(?m)^#{1,6}\s+.*$", text))
        if not matches:
            return chunk_document(doc_id, text, strategy="paragraph")
        bounds: list[tuple[int, int]] = []
        for m_i, m in enumerate(matches):
            seg_start = m.start()
            seg_end = matches[m_i + 1].start() if m_i + 1 < len(matches) else len(text)
            seg_text = text[seg_start:seg_end]
            bounds.append((seg_start, seg_start + len(seg_text.rstrip())))
        segments = bounds

    # merge tiny paragraphs forward
    merged: list[list[int]] = []
    MIN_CHARS = 80
    for s, e in segments:
        if merged and (e - s) < MIN_CHARS:
            merged[-1][1] = e
        elif merged and merged[-1][1] - merged[-1][0] < MIN_CHARS:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    chunks: list[Chunk] = []
    for ordinal, (s, e) in enumerate(merged):
        seg = text[s:e].rstrip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{ordinal}",
                doc_id=doc_id,
                ordinal=ordinal,
                text=seg,
                start=s,
                end=s + len(seg),
                sha=content_hash(seg),
            )
        )
    return chunks


SPAN_FRAMES = ("document", "summary")
"""Coordinate frames a span's ``start``/``end`` may be expressed in.

``document`` — offsets into the original document text named by ``source_id``.
``summary``  — offsets into the text of the summary named by ``source_id``.
"""


def default_summary_id(doc_id: str, level: int, children: list[str]) -> str:
    """Summary id that digests the FULL ordered child list.

    ``store.add_summary`` is ``INSERT OR REPLACE``, so an id derived from only
    the first and last child lets two different sibling sets overwrite each
    other — a parent would then cite a summary whose content had been silently
    replaced. Digesting every child (in order) makes the id collision-free.
    """
    from sherpa.store import content_hash

    return f"sum_{doc_id}_{level}_{len(children)}x{content_hash(chr(10).join(children))[:16]}"


def build_summary(
    store: "Store",
    blob,  # BlobStore; kept for artifact symmetry with kernel usage
    doc_id: str,
    level: int,
    child_ids: list[str],
    summarize: Callable[[str], str],
    summary_id: str | None = None,
) -> str:
    """Create a summary node pointing to EVERY child with exact spans.

    Children may be chunk ids (``doc:N``) or lower-level summary ids.

    Each emitted span is ``{sha, start, end, child_id, of, source_id}`` where
    ``of`` is one of `SPAN_FRAMES` and ``source_id`` names the text those
    offsets index into: the document for ``of="document"``, the cited summary
    for ``of="summary"``. Summary children contribute BOTH a direct pointer
    (so every child is cited) and all of their own spans, so every leaf chunk
    of the DAG stays addressable as an exact *document* span no matter how
    many summary levels sit in between.
    """
    texts: list[str] = []
    spans: list[dict] = []
    for cid in child_ids:
        if ":" in cid and not cid.startswith("sum_"):
            doc_prefix, _ordinal = cid.rsplit(":", 1)
            row = next((r for r in store.get_chunks_by_doc(doc_prefix) if r["chunk_id"] == cid), None)
            if row is None:
                raise KeyError(f"unknown child chunk {cid!r}")
            blob_text = blob.get_text(row["sha"])
            texts.append(blob_text)
            spans.append(
                {"sha": row["sha"], "start": row["start"], "end": row["end"],
                 "child_id": cid, "of": "document", "source_id": row["doc_id"]}
            )
        else:
            child = store.get_summary(cid)
            if child is None:
                raise KeyError(f"unknown child summary {cid!r}")
            texts.append(child["text"])
            from sherpa.store import content_hash

            child_sha = content_hash(child["text"])
            if not blob.exists(child_sha):
                blob.put_text(child["text"])
            direct_pointer = {
                "sha": child_sha, "start": 0, "end": len(child["text"]),
                "child_id": cid, "of": "summary", "source_id": cid,
            }
            transitive_sources = [dict(sp) for sp in child["spans"]]
            spans.append(direct_pointer)
            spans.extend(transitive_sources)
    joined = "\n\n".join(texts)
    summary_text = summarize(joined)
    sid = summary_id or default_summary_id(doc_id, level, child_ids)
    store.add_summary(sid, doc_id, level, summary_text, child_ids, spans)
    return sid


TRUNCATION_NOTICE = "\n[sherpa: summary truncated, {dropped} of {total} characters dropped]"


def bound_summary(text: str, max_chars: int) -> str:
    """Bound *text* to ``max_chars`` of content, recording any loss in-band.

    #492 forbids silently dropping evidence, so an over-long model summary is
    truncated *visibly*: the notice states exactly how much was dropped. The
    notice is appended beyond ``max_chars`` so that the retained summary
    content is never shortened by its own bookkeeping; the result is still
    bounded at ``max_chars + len(notice)``.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_NOTICE.format(
        dropped=len(text) - max_chars, total=len(text)
    )


def summarize_with_channel(
    channel_factory: Callable[[], "ModelChannel"],
    session: str = "summarizer",
    max_chars: int = 1200,
) -> Callable[[str], str]:
    """Build the summarizer callable used by demos; recorded in hermetic runs."""

    def _summarize(text: str) -> str:
        channel = channel_factory()
        resp = channel.complete(
            [
                {"role": "system", "content": f"Summarize in at most {max_chars} characters."},
                {"role": "user", "content": text},
            ],
            session=session,
        )
        return bound_summary(resp.text, max_chars)

    return _summarize


def retrieve(store: "Store", query: str, k: int = 5, doc_prefix: str | None = None) -> list["RetrievalHit"]:
    """FTS-first ranked retrieval (embeddings deliberately deferred)."""
    hits = store.fts_search(query, k=k, doc_prefix=doc_prefix)
    return [
        RetrievalHit(
            chunk_id=h["chunk_id"],
            doc_id=h["doc_id"],
            ordinal=h["ordinal"],
            score=float(h["score"]),
            snippet=h["snippet"],
            sha=h["sha"],
        )
        for h in hits
    ]


class RetrievalHit(BaseModel):
    chunk_id: str
    doc_id: str
    ordinal: int
    score: float
    snippet: str
    sha: str


def scoped_snapshot(store: "Store", run_id: str, kinds=None) -> list:
    """Lock-free read of the subtree's events.

    Descendance is discovered from ``run_started`` payloads carrying
    ``parent_run_id`` (WAL readers never block writers).
    """
    all_events = store.events(run_id=run_id, kinds=kinds)
    starts = store.events(kinds=["run_started"])
    seen_runs = {run_id}
    changed = True
    while changed:
        changed = False
        for ev in starts:
            if ev.run_id in seen_runs:
                continue
            if ev.payload.get("parent_run_id") in seen_runs:
                seen_runs.add(ev.run_id)
                changed = True
    descendants = seen_runs - {run_id}
    out = list(all_events)
    for d in descendants:
        out.extend(store.events(run_id=d, kinds=kinds))
    out.sort(key=lambda e: e.seq or 0)
    return out

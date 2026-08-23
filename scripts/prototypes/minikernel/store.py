"""One memory substrate, four views.

#485 describes four stores: the shared scratchpad, the insight pool, the
context/"inode" tables, and the tool-call history. They need identical
primitives -- content-addressed immutable blobs, an append-only log with
monotonic sequence numbers, a summary tree, and full-text retrieval -- so this
module builds ONE substrate and exposes the four as queries over it.

Design claims under test here:

* C1  Appends never block reads. The #485 protocol ("grab the lock, read the
      tail, *consider whether it alters your plans*, write, release") puts an
      LLM call inside a critical section. Here the lock guards an INSERT only.
* C2  Nothing is ever re-summarised. Segments are sealed at a fixed token
      boundary and summarised exactly once, so summary cost is linear in log
      length rather than quadratic.
* C3  A summary is never the only copy. Every summary records the sequence
      range and content hashes it covers, so it can always be exchanged for
      its source.
* C4  Every context window an agent receives is accompanied by a manifest
      naming every artifact and range that went into it.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# Token accounting. Nothing here needs a real tokeniser to be useful, but the
# estimate must be conservative (over- rather than under-count) so that a
# compiled window never overflows the model it was compiled for.
CHARS_PER_TOKEN = 3.5


def n_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN) + 1)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS blobs(
    hash        TEXT PRIMARY KEY,
    media_type  TEXT NOT NULL,
    n_tokens    INTEGER NOT NULL,
    body        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

-- Append-only. The source of truth. Everything else in the system is a
-- projection of this table.
CREATE TABLE IF NOT EXISTS events(
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    parent_run_id TEXT,
    node_id       TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    type          TEXT NOT NULL,
    session_id    TEXT,
    payload       TEXT NOT NULL,
    created_at    REAL NOT NULL,
    UNIQUE(run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_run    ON events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_events_type   ON events(type);

-- The scratchpad, as an append-only journal. `kind` is the externalised
-- operational record (#485 wants "all thinking"; this stores the parts another
-- agent can act on, not raw chain-of-thought).
CREATE TABLE IF NOT EXISTS notes(
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    node_id    TEXT NOT NULL,
    session_id TEXT,
    kind       TEXT NOT NULL,
    body_hash  TEXT NOT NULL,
    n_tokens   INTEGER NOT NULL,
    scope      TEXT NOT NULL DEFAULT 'run',
    status     TEXT NOT NULL DEFAULT 'accepted',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_kind ON notes(kind, status);

-- Sealed segments and the summary DAG above them. level 0 segments cover raw
-- note ranges; level n>0 segments cover level n-1 segments.
CREATE TABLE IF NOT EXISTS segments(
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    level        INTEGER NOT NULL,
    lo           INTEGER NOT NULL,
    hi           INTEGER NOT NULL,
    covers       TEXT NOT NULL,
    summary_hash TEXT NOT NULL,
    n_tokens     INTEGER NOT NULL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_level ON segments(level, lo);

CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
    body, kind UNINDEXED, seq UNINDEXED, tokenize='porter'
);
CREATE VIRTUAL TABLE IF NOT EXISTS blob_fts USING fts5(
    body, hash UNINDEXED, tokenize='porter'
);
"""


@dataclass
class WindowEntry:
    """One line of a context manifest -- what was supplied, and where from."""

    lane: str
    ref: str
    tokens: int
    body: str


@dataclass
class ContextWindow:
    entries: list[WindowEntry] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(e.tokens for e in self.entries)

    def text(self) -> str:
        return "\n\n".join(f"[{e.lane}:{e.ref}]\n{e.body}" for e in self.entries)

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {"lane": e.lane, "ref": e.ref, "tokens": e.tokens} for e in self.entries
        ]


class Store:
    def __init__(self, path: str, segment_tokens: int = 2000, fanout: int = 8):
        self.path = path
        self.segment_tokens = segment_tokens
        self.fanout = fanout
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()
        # Guards writes only. Readers never take it -- claim C1.
        self._wlock = threading.Lock()

    def close(self) -> None:
        self._db.close()

    # ---------------------------------------------------------------- blobs

    def put_blob(self, body: str, media_type: str = "text/plain") -> str:
        h = content_hash(body)
        with self._wlock:
            self._db.execute(
                "INSERT OR IGNORE INTO blobs(hash, media_type, n_tokens, body,"
                " created_at) VALUES(?,?,?,?,?)",
                (h, media_type, n_tokens(body), body, time.time()),
            )
            self._db.execute(
                "INSERT INTO blob_fts(body, hash) VALUES(?,?)", (body, h)
            )
            self._db.commit()
        return h

    def get_blob(self, h: str) -> str | None:
        row = self._db.execute("SELECT body FROM blobs WHERE hash=?", (h,)).fetchone()
        return row["body"] if row else None

    # --------------------------------------------------------------- events

    def append_event(
        self,
        run_id: str,
        node_id: str,
        type_: str,
        payload: dict[str, Any] | None = None,
        parent_run_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        body = json.dumps(payload or {}, sort_keys=True, default=str)
        with self._wlock:
            seq = self._db.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
            self._db.execute(
                "INSERT INTO events(run_id, parent_run_id, node_id, seq, type,"
                " session_id, payload, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (run_id, parent_run_id, node_id, seq, type_, session_id, body,
                 time.time()),
            )
            self._db.commit()
        return seq

    def events(self, root_run_id: str, descendants: bool = True) -> list[sqlite3.Row]:
        if not descendants:
            return list(
                self._db.execute(
                    "SELECT * FROM events WHERE run_id=? ORDER BY seq", (root_run_id,)
                )
            )
        return list(
            self._db.execute(
                "WITH RECURSIVE sub(r) AS ("
                "  SELECT ?"
                "  UNION"
                "  SELECT e.run_id FROM events e JOIN sub s ON e.parent_run_id = s.r"
                ") SELECT e.* FROM events e JOIN sub ON e.run_id = sub.r"
                " ORDER BY e.run_id, e.seq",
                (root_run_id,),
            )
        )

    # ---------------------------------------------------- journal (scratchpad)

    def append_note(
        self,
        run_id: str,
        node_id: str,
        kind: str,
        body: str,
        session_id: str | None = None,
        scope: str = "run",
        status: str = "accepted",
    ) -> int:
        """Append to the shared journal. O(1), no deliberation under lock."""
        h = self.put_blob(body, "text/note")
        with self._wlock:
            cur = self._db.execute(
                "INSERT INTO notes(run_id, node_id, session_id, kind, body_hash,"
                " n_tokens, scope, status, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, node_id, session_id, kind, h, n_tokens(body), scope,
                 status, time.time()),
            )
            seq = cur.lastrowid
            self._db.execute(
                "INSERT INTO note_fts(body, kind, seq) VALUES(?,?,?)", (body, kind, seq)
            )
            self._db.commit()
        return int(seq)

    def note(self, seq: int) -> sqlite3.Row | None:
        return self._db.execute("SELECT * FROM notes WHERE seq=?", (seq,)).fetchone()

    def note_body(self, seq: int) -> str:
        row = self.note(seq)
        return self.get_blob(row["body_hash"]) or "" if row else ""

    def notes_tail(self, budget_tokens: int) -> list[sqlite3.Row]:
        """Most recent notes that fit, newest-first walk, returned oldest-first."""
        out: list[sqlite3.Row] = []
        used = 0
        for row in self._db.execute("SELECT * FROM notes ORDER BY seq DESC"):
            if used + row["n_tokens"] > budget_tokens:
                break
            out.append(row)
            used += row["n_tokens"]
        return list(reversed(out))

    # ----------------------------------------------------------- summary DAG

    def unsealed_span(self) -> tuple[int, int, int]:
        """(lo, hi, tokens) of notes not yet covered by a level-0 segment."""
        row = self._db.execute(
            "SELECT COALESCE(MAX(hi), 0) FROM segments WHERE level=0"
        ).fetchone()
        lo = int(row[0]) + 1
        agg = self._db.execute(
            "SELECT COALESCE(MAX(seq),0), COALESCE(SUM(n_tokens),0) FROM notes"
            " WHERE seq >= ?",
            (lo,),
        ).fetchone()
        return lo, int(agg[0]), int(agg[1])

    def seal(self, summarize: Callable[[str, int], str]) -> list[int]:
        """Seal every complete segment. Each range is summarised exactly once.

        `summarize(text, level)` returns the summary. Sealing is idempotent:
        calling it twice with no new notes does nothing (claim C2).
        """
        created: list[int] = []
        # level 0: raw notes -> segment summaries
        while True:
            lo, hi, tokens = self.unsealed_span()
            if tokens < self.segment_tokens or hi < lo:
                break
            span_lo, acc, span_hi = lo, 0, lo - 1
            for row in self._db.execute(
                "SELECT seq, n_tokens FROM notes WHERE seq >= ? ORDER BY seq", (lo,)
            ):
                acc += row["n_tokens"]
                span_hi = row["seq"]
                if acc >= self.segment_tokens:
                    break
            bodies, hashes = [], []
            for row in self._db.execute(
                "SELECT seq, kind, body_hash FROM notes WHERE seq BETWEEN ? AND ?"
                " ORDER BY seq",
                (span_lo, span_hi),
            ):
                hashes.append(row["body_hash"])
                bodies.append(f"({row['seq']}/{row['kind']}) {self.get_blob(row['body_hash'])}")
            created.append(
                self._insert_segment(0, span_lo, span_hi, hashes,
                                     summarize("\n".join(bodies), 0))
            )
        # levels 1..n: segments -> higher summaries
        level = 0
        while True:
            rows = list(
                self._db.execute(
                    "SELECT * FROM segments WHERE level=? ORDER BY lo", (level,)
                )
            )
            higher = {
                c
                for r in self._db.execute(
                    "SELECT covers FROM segments WHERE level=?", (level + 1,)
                )
                for c in json.loads(r["covers"])
            }
            pending = [r for r in rows if str(r["id"]) not in higher]
            if len(pending) < self.fanout:
                break
            for i in range(0, len(pending) - self.fanout + 1, self.fanout):
                group = pending[i : i + self.fanout]
                text = "\n".join(self.get_blob(g["summary_hash"]) or "" for g in group)
                created.append(
                    self._insert_segment(
                        level + 1,
                        group[0]["lo"],
                        group[-1]["hi"],
                        [str(g["id"]) for g in group],
                        summarize(text, level + 1),
                    )
                )
            level += 1
        return created

    def _insert_segment(
        self, level: int, lo: int, hi: int, covers: Sequence[str], summary: str
    ) -> int:
        h = self.put_blob(summary, "text/summary")
        with self._wlock:
            cur = self._db.execute(
                "INSERT INTO segments(level, lo, hi, covers, summary_hash, n_tokens,"
                " created_at) VALUES(?,?,?,?,?,?,?)",
                (level, lo, hi, json.dumps(list(covers)), h, n_tokens(summary),
                 time.time()),
            )
            self._db.commit()
        return int(cur.lastrowid)

    def top_segments(self) -> list[sqlite3.Row]:
        """Highest-level segments plus any lower ones they do not cover."""
        rows = list(self._db.execute("SELECT * FROM segments"))
        covered: set[str] = set()
        for r in rows:
            if r["level"] > 0:
                covered |= set(json.loads(r["covers"]))
        return sorted(
            (r for r in rows if str(r["id"]) not in covered), key=lambda r: (r["lo"],)
        )

    def expand(self, segment_id: int) -> list[str]:
        """Exchange a summary for what it covers -- claim C3."""
        row = self._db.execute(
            "SELECT * FROM segments WHERE id=?", (segment_id,)
        ).fetchone()
        if row is None:
            return []
        covers = json.loads(row["covers"])
        if row["level"] == 0:
            return [self.get_blob(h) or "" for h in covers]
        out = []
        for cid in covers:
            child = self._db.execute(
                "SELECT summary_hash FROM segments WHERE id=?", (int(cid),)
            ).fetchone()
            if child:
                out.append(self.get_blob(child["summary_hash"]) or "")
        return out

    # ------------------------------------------------------------ retrieval

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = [t for t in re.findall(r"[A-Za-z0-9_]+", query) if len(t) > 2]
        return " OR ".join(f'"{t}"' for t in terms[:24])

    def search_notes(self, query: str, k: int = 8, kind: str | None = None):
        q = self._fts_query(query)
        if not q:
            return []
        sql = (
            "SELECT n.*, bm25(note_fts) AS score FROM note_fts"
            " JOIN notes n ON n.seq = note_fts.seq"
            " WHERE note_fts MATCH ?"
        )
        args: list[Any] = [q]
        if kind:
            sql += " AND n.kind = ?"
            args.append(kind)
        sql += " ORDER BY score LIMIT ?"
        args.append(k)
        try:
            return list(self._db.execute(sql, args))
        except sqlite3.OperationalError:
            return []

    # ------------------------------------------------- the four #485 views

    def view_scratchpad(self, limit: int = 200):
        return list(
            self._db.execute("SELECT * FROM notes ORDER BY seq DESC LIMIT ?", (limit,))
        )

    def view_insights(self):
        return list(
            self._db.execute(
                "SELECT * FROM notes WHERE kind='insight' AND status='accepted'"
                " ORDER BY seq"
            )
        )

    def view_tool_history(self, capability: str | None = None):
        rows = self._db.execute(
            "SELECT * FROM events WHERE type='capability_invoked' ORDER BY id"
        )
        out = [dict(r) for r in rows]
        if capability:
            out = [r for r in out if json.loads(r["payload"]).get("capability") == capability]
        return out

    def view_summary_tree(self):
        return list(self._db.execute("SELECT * FROM segments ORDER BY level, lo"))

    # --------------------------------------------------- context compilation

    def compile_window(
        self,
        budget_tokens: int,
        query: str = "",
        lanes: dict[str, float] | None = None,
    ) -> ContextWindow:
        """Compile a context window under a per-lane token budget (claim C4).

        Lane fractions are POLICY, not constants: a deterministic leaf can ask
        for {'tail': 1.0} and a planner for {'insights': .5, 'retrieved': .5}.
        """
        lanes = lanes or {"tail": 0.35, "summaries": 0.25, "retrieved": 0.25,
                          "insights": 0.15}
        win = ContextWindow()
        seen: set[str] = set()

        def add(lane: str, ref: str, body: str, cap: int) -> bool:
            t = n_tokens(body)
            if ref in seen:
                return False
            if t > cap:
                win.dropped.append(ref)
                return False
            seen.add(ref)
            win.entries.append(WindowEntry(lane, ref, t, body))
            return True

        for lane, frac in lanes.items():
            cap = int(budget_tokens * frac)
            used = 0
            if lane == "tail":
                for row in self.notes_tail(cap):
                    b = self.get_blob(row["body_hash"]) or ""
                    if used + n_tokens(b) > cap:
                        break
                    if add(lane, f"note:{row['seq']}", b, cap - used):
                        used += n_tokens(b)
            elif lane == "summaries":
                for seg in self.top_segments():
                    b = self.get_blob(seg["summary_hash"]) or ""
                    if used + n_tokens(b) > cap:
                        win.dropped.append(f"seg:{seg['id']}")
                        continue
                    if add(lane, f"seg:{seg['id']}", b, cap - used):
                        used += n_tokens(b)
            elif lane == "retrieved" and query:
                for row in self.search_notes(query, k=12):
                    b = self.get_blob(row["body_hash"]) or ""
                    if used + n_tokens(b) > cap:
                        break
                    if add(lane, f"note:{row['seq']}", b, cap - used):
                        used += n_tokens(b)
            elif lane == "insights":
                cands = self.search_notes(query, k=8, kind="insight") if query else []
                if not cands:
                    cands = self.view_insights()[-8:]
                for row in cands:
                    b = self.get_blob(row["body_hash"]) or ""
                    if used + n_tokens(b) > cap:
                        break
                    if add(lane, f"insight:{row['seq']}", b, cap - used):
                        used += n_tokens(b)
        return win

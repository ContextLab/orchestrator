"""Durable single-machine storage for sherpa (issue #492, MVP scope item 2).

SQLite in WAL mode holds the append-only event log — the source of truth —
plus deterministic projections built transactionally alongside appends.
Large immutable artifacts live in a content-addressed filesystem blob store.
FTS5 indexes structural chunks for retrieval; the solution cache stores
positive AND negative outcomes, and refuses to record budget exhaustion as
evidence against a plan.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from sherpa.events import EVENT_KINDS, Event
from sherpa.ir import TERMINAL_STATES

#: Runs of word characters — the only part of a user query FTS5 can tokenize.
_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

FINDING_DISPOSITIONS = ("open", "fixed", "accepted_risk", "invalid", "deferred", "superseded")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run_id TEXT NOT NULL,
    node_key TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    causal_seq INTEGER
);
CREATE INDEX IF NOT EXISTS ix_events_run ON events(run_id, seq);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    problem_sha TEXT,
    parent_run_id TEXT,
    plan_sha TEXT,
    error TEXT,
    created_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    run_id TEXT NOT NULL,
    node_key TEXT NOT NULL,
    state TEXT NOT NULL,
    owner_session TEXT,
    depth INTEGER NOT NULL DEFAULT 0,
    parent_key TEXT,
    updated_ts REAL NOT NULL,
    PRIMARY KEY (run_id, node_key)
);
CREATE TABLE IF NOT EXISTS leases (
    run_id TEXT NOT NULL,
    node_key TEXT NOT NULL,
    session TEXT NOT NULL,
    expires_ts REAL NOT NULL,
    PRIMARY KEY (run_id, node_key)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    to_node_key TEXT NOT NULL,
    sender TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    delivered INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    run_id TEXT PRIMARY KEY,
    tokens REAL NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    nodes INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    wall_seconds REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS solution_cache (
    signature TEXT PRIMARY KEY,
    entry TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, chunk_id UNINDEXED, doc_id UNINDEXED
);
CREATE TABLE IF NOT EXISTS chunks_meta (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    sha TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    summary_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    level INTEGER NOT NULL,
    text TEXT NOT NULL,
    children TEXT NOT NULL,
    spans TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    criterion TEXT NOT NULL,
    evidence_ref TEXT NOT NULL DEFAULT '',
    blocking INTEGER NOT NULL DEFAULT 0,
    disposition TEXT NOT NULL DEFAULT 'open',
    rationale TEXT NOT NULL DEFAULT '',
    created_ts REAL NOT NULL
);
"""


def content_hash(data: bytes | str) -> str:
    """sha256 hex digest of *data* (strings encoded utf-8)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    """Content-addressed immutable artifact store on the filesystem."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)

    def path(self, sha: str) -> Path:
        return self.root / "objects" / sha[:2] / sha

    def exists(self, sha: str) -> bool:
        return self.path(sha).exists()

    def put_bytes(self, data: bytes) -> str:
        sha = content_hash(data)
        dest = self.path(sha)
        if not dest.exists():  # immutable; write-once
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(dest)
        return sha

    def put_text(self, text: str) -> str:
        return self.put_bytes(text.encode("utf-8"))

    def get_bytes(self, sha: str) -> bytes:
        p = self.path(sha)
        if not p.exists():
            raise KeyError(f"unknown blob {sha[:12]}")
        return p.read_bytes()

    def get_text(self, sha: str) -> str:
        return self.get_bytes(sha).decode("utf-8")


class Store:
    """Event log + projections + retrieval substrate over one SQLite file."""

    def __init__(self, path: Path, blob_root: Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.blob = BlobStore(blob_root or self.path.parent / "blobs")
        self.conn = sqlite3.connect(str(self.path), timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        assert self.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _j(data: Any) -> str:
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _uj(raw: str | None) -> Any:
        return json.loads(raw) if raw else {}

    def _log(
        self,
        kind: str,
        run_id: str,
        node_key: str | None = None,
        payload: dict[str, Any] | None = None,
        cursor: sqlite3.Cursor | None = None,
    ) -> Event:
        event = Event(kind=kind, run_id=run_id, node_key=node_key, payload=payload or {})
        params = (event.ts, event.run_id, event.node_key, event.kind, self._j(event.payload))
        executor = cursor if cursor is not None else self.conn
        row = executor.execute(
            "INSERT INTO events (ts, run_id, node_key, kind, payload) VALUES (?,?,?,?,?)",
            params,
        )
        event.seq = int(row.lastrowid)
        if cursor is None:
            self.conn.commit()
        return event

    # -- event log -----------------------------------------------------------

    def append(self, event: Event) -> Event:
        """Append *event* to the log; validates its kind and assigns a seq."""
        if event.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {event.kind!r}")
        cur = self.conn.execute(
            "INSERT INTO events (ts, run_id, node_key, kind, payload, causal_seq) VALUES (?,?,?,?,?,?)",
            (
                event.ts,
                event.run_id,
                event.node_key,
                event.kind,
                self._j(event.payload),
                event.causal_seq,
            ),
        )
        event.seq = int(cur.lastrowid)
        self._project_event(event)
        self.conn.commit()
        return event

    def _project_event(self, event: Event, *, cursor: sqlite3.Cursor | None = None) -> None:
        c = cursor or self.conn
        now = event.ts
        p = event.payload
        if event.kind == "run_started":
            c.execute(
                "INSERT OR IGNORE INTO runs (run_id, status, problem_sha, parent_run_id, plan_sha, created_ts)"
                " VALUES (?,?,?,?,?,?)",
                (
                    event.run_id,
                    p.get("status", "running"),
                    p.get("problem_sha"),
                    p.get("parent_run_id"),
                    p.get("plan_sha"),
                    now,
                ),
            )
        elif event.kind == "run_terminal":
            c.execute(
                "UPDATE runs SET status=?, error=? WHERE run_id=?",
                (p.get("status"), p.get("error"), event.run_id),
            )
        elif event.kind == "node_created":
            c.execute(
                "INSERT OR IGNORE INTO nodes (run_id, node_key, state, depth, parent_key, updated_ts)"
                " VALUES (?,?,?,?,?,?)",
                (event.run_id, event.node_key, p.get("state", "pending"), p.get("depth", 0), p.get("parent_key"), now),
            )
        elif event.kind == "node_state_changed":
            c.execute(
                "UPDATE nodes SET state=?, owner_session=?, updated_ts=? WHERE run_id=? AND node_key=?",
                (p.get("new"), p.get("owner_session"), now, event.run_id, event.node_key),
            )

    def events(
        self,
        *,
        run_id: str | None = None,
        kinds: Iterable[str] | None = None,
        since_seq: int = 0,
        limit: int | None = None,
    ) -> list[Event]:
        q = "SELECT seq, ts, run_id, node_key, kind, payload, causal_seq FROM events WHERE seq > ?"
        args: list[Any] = [since_seq]
        if run_id is not None:
            q += " AND run_id = ?"
            args.append(run_id)
        if kinds is not None:
            kinds = list(kinds)
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        q += " ORDER BY seq"
        if limit is not None:
            q += " LIMIT ?"
            args.append(limit)
        rows = self.conn.execute(q, args).fetchall()
        return [
            Event(
                seq=r["seq"],
                ts=r["ts"],
                run_id=r["run_id"],
                node_key=r["node_key"],
                kind=r["kind"],
                payload=self._uj(r["payload"]),
                causal_seq=r["causal_seq"],
            )
            for r in rows
        ]

    def head_seq(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM events").fetchone()
        return int(row["m"])

    # -- runs & nodes ----------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        problem_sha: str,
        status: str = "running",
        parent_run_id: str | None = None,
        plan_sha: str | None = None,
    ) -> None:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, status, problem_sha, parent_run_id, plan_sha, created_ts)"
            " VALUES (?,?,?,?,?,?)",
            (run_id, status, problem_sha, parent_run_id, plan_sha, time.time()),
        )
        self._log(
            "run_started",
            run_id,
            payload={"status": status, "problem_sha": problem_sha, "parent_run_id": parent_run_id},
        )
        self.conn.commit()
        assert cur.rowcount >= 0

    def set_run_status(self, run_id: str, status: str, error: str | None = None) -> None:
        if status not in TERMINAL_STATES and status not in ("running", "paused"):
            raise ValueError(f"invalid run status {status!r}")
        self.conn.execute("UPDATE runs SET status=?, error=? WHERE run_id=?", (status, error, run_id))
        if status in TERMINAL_STATES:
            self._log("run_terminal", run_id, payload={"status": status, "error": error})
        self.conn.commit()

    def upsert_node(
        self,
        run_id: str,
        node_key: str,
        state: str = "pending",
        depth: int = 0,
        parent_key: str | None = None,
    ) -> None:
        cur = self.conn.execute(
            "INSERT INTO nodes (run_id, node_key, state, depth, parent_key, updated_ts) VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(run_id, node_key) DO UPDATE SET state=excluded.state,"
            " depth=excluded.depth, parent_key=excluded.parent_key, updated_ts=excluded.updated_ts",
            (run_id, node_key, state, depth, parent_key, time.time()),
        )
        if cur.rowcount == 1:
            self._log(
                "node_created",
                run_id,
                node_key=node_key,
                payload={"state": state, "depth": depth, "parent_key": parent_key},
            )
        else:
            self._log("node_state_changed", run_id, node_key=node_key, payload={"new": state})
        self.conn.commit()

    def cas_node_state(
        self,
        run_id: str,
        node_key: str,
        expected: str,
        new: str,
        owner_session: str | None = None,
    ) -> bool:
        """Compare-and-swap the node state; the only writer of transitions."""
        cur = self.conn.execute(
            # COALESCE: callers that only change state (e.g. running ->
            # completed) must not blank the owning session.
            "UPDATE nodes SET state=?, owner_session=COALESCE(?, owner_session), updated_ts=?"
            " WHERE run_id=? AND node_key=? AND state=?",
            (new, owner_session, time.time(), run_id, node_key, expected),
        )
        ok = cur.rowcount == 1
        if ok:
            self._log(
                "node_state_changed",
                run_id,
                node_key=node_key,
                payload={"expected": expected, "new": new, "owner_session": owner_session},
            )
        self.conn.commit()
        return ok

    # -- leases ----------------------------------------------------------------

    def acquire_lease(
        self,
        run_id: str,
        node_key: str,
        session: str,
        ttl_s: float = 120.0,
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        with self.conn:
            row = self.conn.execute(
                "SELECT session, expires_ts FROM leases WHERE run_id=? AND node_key=?",
                (run_id, node_key),
            ).fetchone()
            if row is not None and row["expires_ts"] > now and row["session"] != session:
                return False  # held by a *different* live session
            self.conn.execute(
                "INSERT OR REPLACE INTO leases (run_id, node_key, session, expires_ts) VALUES (?,?,?,?)",
                (run_id, node_key, session, now + ttl_s),
            )
        self._log("lease_acquired", run_id, node_key=node_key, payload={"session": session, "ttl_s": ttl_s})
        self.conn.commit()
        return True

    def renew_lease(self, run_id: str, node_key: str, session: str, ttl_s: float = 120.0) -> bool:
        cur = self.conn.execute(
            "UPDATE leases SET expires_ts=? WHERE run_id=? AND node_key=? AND session=?",
            (time.time() + ttl_s, run_id, node_key, session),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def release_lease(self, run_id: str, node_key: str, session: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM leases WHERE run_id=? AND node_key=? AND session=?",
            (run_id, node_key, session),
        )
        self.conn.commit()
        if cur.rowcount == 1:
            self._log("lease_released", run_id, node_key=node_key, payload={"session": session})
            self.conn.commit()
        return cur.rowcount == 1

    def expired_leases(self, now: float | None = None) -> list[tuple[str, str, str]]:
        at = time.time() if now is None else now
        rows = self.conn.execute("SELECT run_id, node_key, session FROM leases WHERE expires_ts <= ?", (at,)).fetchall()
        return [(r["run_id"], r["node_key"], r["session"]) for r in rows]

    # -- messages ----------------------------------------------------------------

    def enqueue_message(self, run_id: str, to_node_key: str, payload: dict, sender: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages (run_id, to_node_key, sender, payload) VALUES (?,?,?,?)",
            (run_id, to_node_key, sender, self._j(payload)),
        )
        self._log("message_enqueued", run_id, node_key=to_node_key, payload={"payload": payload, "sender": sender})
        self.conn.commit()
        return int(cur.lastrowid)

    def take_messages(self, run_id: str, node_key: str) -> list[dict]:
        """Deliver pending messages exactly once (checkpoint-boundary semantics)."""
        out: list[dict] = []
        with self.conn:
            rows = self.conn.execute(
                "SELECT id, payload FROM messages WHERE run_id=? AND to_node_key=? AND delivered=0 ORDER BY id",
                (run_id, node_key),
            ).fetchall()
            for r in rows:
                got = self.conn.execute(
                    "UPDATE messages SET delivered=1 WHERE id=? AND delivered=0",
                    (r["id"],),
                )
                if got.rowcount == 1:
                    payload = self._uj(r["payload"])
                    out.append(payload)
                    self._log("message_delivered", run_id, node_key=node_key, payload={"payload": payload}, cursor=None)
        self.conn.commit()
        return out

    # -- usage / budgets -----------------------------------------------------------

    def add_usage(self, run_id: str, **deltas: float) -> None:
        cols = {
            "tokens": "tokens",
            "cost_usd": "cost_usd",
            "nodes": "nodes",
            "attempts": "attempts",
            "wall_seconds": "wall_seconds",
        }
        sets = []
        args: list[float] = []
        for k, v in deltas.items():
            if k not in cols:
                raise ValueError(f"unknown usage field {k!r}")
            sets.append(f"{cols[k]} = {cols[k]} + ?")
            args.append(float(v))
        self.conn.execute("INSERT OR IGNORE INTO usage (run_id) VALUES (?)", (run_id,))
        if sets:
            self.conn.execute(
                f"UPDATE usage SET {', '.join(sets)} WHERE run_id=?",  # noqa: S608 - fixed column names
                (*args, run_id),
            )
        self._log("usage_checkpoint", run_id, payload={k: float(v) for k, v in deltas.items()})
        self.conn.commit()

    def usage(self, run_id: str) -> dict[str, float]:
        row = self.conn.execute("SELECT * FROM usage WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return {"tokens": 0.0, "cost_usd": 0.0, "nodes": 0, "attempts": 0, "wall_seconds": 0.0}
        return {
            "tokens": row["tokens"],
            "cost_usd": row["cost_usd"],
            "nodes": row["nodes"],
            "attempts": row["attempts"],
            "wall_seconds": row["wall_seconds"],
        }

    # -- solution cache ---------------------------------------------------------------

    def cache_get(self, signature: str) -> dict | None:
        row = self.conn.execute("SELECT entry FROM solution_cache WHERE signature=?", (signature,)).fetchone()
        if row is None:
            return None
        entry = self._uj(row["entry"])
        self._log("cache_hit", str(entry.get("run_id")), payload={"signature": signature}) if entry.get("run_id") else None
        return entry

    def cache_put(self, signature: str, entry: dict) -> None:
        if entry.get("status_class") == "failed" and entry.get("inconclusive"):
            raise ValueError("budget exhaustion must not be recorded as evidence against a plan")
        self.conn.execute(
            "INSERT OR REPLACE INTO solution_cache (signature, entry) VALUES (?,?)",
            (signature, self._j(entry)),
        )
        self.conn.commit()

    # -- FTS5 chunks + summaries --------------------------------------------------------

    def index_chunk(self, chunk: dict) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks_meta (chunk_id, doc_id, ordinal, start, end, sha)"
                " VALUES (?,?,?,?,?,?)",
                (chunk["chunk_id"], chunk["doc_id"], chunk["ordinal"], chunk["start"], chunk["end"], chunk["sha"]),
            )
            self.conn.execute(
                "INSERT INTO chunks_fts (text, chunk_id, doc_id) VALUES (?,?,?)",
                (chunk["text"], chunk["chunk_id"], chunk["doc_id"]),
            )
        self._log("chunk_indexed", str(chunk.get("run_id", "")), payload={"chunk_id": chunk["chunk_id"]})
        self.conn.commit()

    @staticmethod
    def _fts_match(query: str) -> str | None:
        """Render *query* as a MATCH expression of literal, quoted terms.

        FTS5 MATCH is a query *language*: bare user text can carry column
        filters (``foo:bar``), prefix/special syntax (``*``), unbalanced quotes
        and operators, each of which raises ``sqlite3.OperationalError`` and
        would fail the calling node. Every token is therefore quoted, which is
        the only form FTS5 treats as data rather than syntax. Terms are joined
        by whitespace: implicit AND, matching prior behaviour for plain text.
        Returns ``None`` when nothing searchable remains.
        """
        tokens = _FTS_TOKEN_RE.findall(query)
        if not tokens:
            return None
        return " ".join('"%s"' % t.replace('"', '""') for t in tokens)

    def fts_search(self, query: str, k: int = 5, doc_prefix: str | None = None) -> list[dict]:
        sql = (
            "SELECT cm.chunk_id, cm.doc_id, cm.ordinal, cm.start, cm.end, cm.sha,"
            " bm25(chunks_fts) AS score, snippet(chunks_fts, 0, '<', '>', '…', 12) AS snip"
            " FROM chunks_fts JOIN chunks_meta cm ON cm.chunk_id = chunks_fts.chunk_id"
            " WHERE chunks_fts MATCH ?"
        )
        match = self._fts_match(query)
        if match is None:
            return []  # no searchable term survived tokenization
        args: list[Any] = [match]
        if doc_prefix is not None:
            sql += " AND cm.doc_id LIKE ?"
            args.append(doc_prefix + "%")
        sql += " ORDER BY score LIMIT ?"
        args.append(k)
        rows = self.conn.execute(sql, args).fetchall()
        return [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "ordinal": r["ordinal"],
                "start": r["start"],
                "end": r["end"],
                "sha": r["sha"],
                "score": r["score"],
                "snippet": r["snip"],
            }
            for r in rows
        ]

    def get_chunks_by_doc(self, doc_id: str) -> list[dict]:
        """All chunks of one document in ordinal order (no FTS query involved)."""
        rows = self.conn.execute(
            "SELECT cm.chunk_id, cm.doc_id, cm.ordinal, cm.start, cm.end, cm.sha, cf.text"
            " FROM chunks_meta cm JOIN chunks_fts cf ON cf.chunk_id = cm.chunk_id"
            " WHERE cm.doc_id = ? ORDER BY cm.ordinal",
            (doc_id,),
        ).fetchall()
        return [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "ordinal": r["ordinal"],
                "start": r["start"],
                "end": r["end"],
                "sha": r["sha"],
                "text": r["text"],
            }
            for r in rows
        ]

    def add_summary(
        self,
        summary_id: str,
        doc_id: str,
        level: int,
        text: str,
        children: list[str],
        spans: list[dict],
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO summaries (summary_id, doc_id, level, text, children, spans)"
            " VALUES (?,?,?,?,?,?)",
            (summary_id, doc_id, level, text, self._j(children), self._j(spans)),
        )
        self._log("summary_created", doc_id, payload={"summary_id": summary_id, "level": level, "n_children": len(children)})
        self.conn.commit()

    def get_summary(self, summary_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM summaries WHERE summary_id=?", (summary_id,)).fetchone()
        if row is None:
            return None
        return {
            "summary_id": row["summary_id"],
            "doc_id": row["doc_id"],
            "level": row["level"],
            "text": row["text"],
            "children": self._uj(row["children"]),
            "spans": self._uj(row["spans"]),
        }

    # -- findings ------------------------------------------------------------------

    def add_finding(self, finding: dict) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO findings (finding_id, subject, criterion, evidence_ref, blocking,"
            " disposition, rationale, created_ts) VALUES (?,?,?,?,?,?,?,?)",
            (
                finding["id"],
                finding["subject"],
                finding["criterion"],
                finding.get("evidence_ref", ""),
                1 if finding.get("blocking") else 0,
                finding.get("disposition", "open"),
                finding.get("rationale", ""),
                time.time(),
            ),
        )
        self._log(
            "finding_raised",
            str(finding.get("run_id", "")),
            payload={
                "finding_id": finding["id"],
                "subject": finding["subject"],
                "blocking": bool(finding.get("blocking")),
            },
        )
        self.conn.commit()

    def set_finding_disposition(self, finding_id: str, disposition: str, rationale: str = "") -> bool:
        if disposition not in FINDING_DISPOSITIONS:
            raise ValueError(f"invalid disposition {disposition!r}")
        cur = self.conn.execute(
            "UPDATE findings SET disposition=?, rationale=? WHERE finding_id=?",
            (disposition, rationale, finding_id),
        )
        self.conn.commit()
        if cur.rowcount == 1:
            self._log("finding_disposition", "", payload={"finding_id": finding_id, "disposition": disposition})
            self.conn.commit()
        return cur.rowcount == 1

    def findings(self, subject: str | None = None) -> list[dict]:
        if subject is None:
            rows = self.conn.execute("SELECT * FROM findings ORDER BY created_ts").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM findings WHERE subject=? ORDER BY created_ts", (subject,)
            ).fetchall()
        return [
            {
                "id": r["finding_id"],
                "subject": r["subject"],
                "criterion": r["criterion"],
                "evidence_ref": r["evidence_ref"],
                "blocking": bool(r["blocking"]),
                "disposition": r["disposition"],
                "rationale": r["rationale"],
            }
            for r in rows
        ]

    # -- projections -----------------------------------------------------------------

    def projection_node_state(self, run_id: str, node_key: str) -> str | None:
        row = self.conn.execute(
            "SELECT state FROM nodes WHERE run_id=? AND node_key=?",
            (run_id, node_key),
        ).fetchone()
        return row["state"] if row else None

    def projection(self, run_id: str) -> dict:
        run = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        nodes_rows = self.conn.execute("SELECT * FROM nodes WHERE run_id=? ORDER BY node_key", (run_id,)).fetchall()
        usage = self.usage(run_id)
        pending = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE run_id=? AND delivered=0", (run_id,)
        ).fetchone()["n"]
        finds = self.findings()
        return {
            "run_id": run_id,
            "status": run["status"] if run else None,
            "error": run["error"] if run else None,
            "parent_run_id": run["parent_run_id"] if run else None,
            "nodes": {
                r["node_key"]: {
                    "state": r["state"],
                    "owner_session": r["owner_session"],
                    "depth": r["depth"],
                    "parent_key": r["parent_key"],
                }
                for r in nodes_rows
            },
            "usage": usage,
            "messages_pending": pending,
            "findings": sorted(
                f["id"] for f in finds if f.get("subject", "").startswith(run_id)
            ),
        }

    def replay_projection(self, run_id: str) -> dict:
        """Rebuild the projection purely from the event log (resume-by-replay basis)."""
        nodes: dict[str, dict] = {}
        status = None
        error = None
        parent_run_id = None
        usage_totals = {"tokens": 0.0, "cost_usd": 0.0, "nodes": 0, "attempts": 0, "wall_seconds": 0.0}
        messages_pending = 0
        for ev in self.events(run_id=run_id):
            if ev.kind == "run_started":
                status = ev.payload.get("status", "running")
                parent_run_id = ev.payload.get("parent_run_id")
            elif ev.kind == "run_terminal":
                status = ev.payload.get("status")
                error = ev.payload.get("error")
            elif ev.kind == "node_created":
                nodes[ev.node_key] = {  # type: ignore[index]
                    "state": ev.payload.get("state", "pending"),
                    "owner_session": None,
                    "depth": ev.payload.get("depth", 0),
                    "parent_key": ev.payload.get("parent_key"),
                }
            elif ev.kind == "node_state_changed":
                key = ev.node_key
                if key in nodes:
                    nodes[key]["state"] = ev.payload.get("new", nodes[key]["state"])  # type: ignore[index]
                    # Mirrors the live COALESCE: a transition that does not
                    # name a session must not erase the recorded owner.
                    owner = ev.payload.get("owner_session")
                    if owner is not None:
                        nodes[key]["owner_session"] = owner  # type: ignore[index]
            elif ev.kind == "usage_checkpoint":
                for field, delta in ev.payload.items():
                    usage_totals[field] = usage_totals.get(field, 0.0) + float(delta)
            elif ev.kind == "message_enqueued":
                messages_pending += 1
            elif ev.kind == "message_delivered":
                messages_pending = max(0, messages_pending - 1)
        # Findings are scoped by subject prefix, exactly as :meth:`projection` does;
        # ``finding_raised`` carries both the id and the subject for this reason.
        finding_ids = {
            str(ev.payload["finding_id"])
            for ev in self.events(kinds=["finding_raised"])
            if ev.payload.get("finding_id") is not None
            and str(ev.payload.get("subject", "")).startswith(run_id)
        }
        return {
            "run_id": run_id,
            "status": status,
            "error": error,
            "parent_run_id": parent_run_id,
            "nodes": nodes,
            "usage": usage_totals,
            "messages_pending": messages_pending,
            "findings": sorted(finding_ids),
        }

    def close(self) -> None:
        self.conn.close()

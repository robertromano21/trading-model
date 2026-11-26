"""Write an item into the AI analyst's memory from Python (e.g. the paper trader's decision / chat-log writer).

Rows land in <data dir>/ai_memory.sqlite3 (gitignored) table memory_items with embedded=0; the Node analyst
embeds them with the local MiniLM model on its next pass (within ~60 s while the dashboard runs, or on the next
recall). Dedupe uses the same hash as server/ai/memory/memory.js, so re-sending the same decision is a no-op.

    from memory_ingest import ingest
    ingest("DECISION", "BUY 12 NVDA @ 151.20 (tsmom sleeve, target 18%); advisory YELLOW -> half size",
           ticker="NVDA", ts="2026-11-25T21:30:00-05:00",
           meta={"refPrice": 151.2, "action": "BUY", "shares": 12, "account": "default", "dedupeKey": "order:123"})

Kinds: QA, CHAT, VERDICT, DECISION, NEWS, FUNDAMENTALS, NOTE, EVENT. meta.refPrice enables "then/now" outcome notes.
"""
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

KINDS = {"QA", "CHAT", "VERDICT", "DECISION", "NEWS", "FUNDAMENTALS", "NOTE", "EVENT"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, ticker TEXT, ts TEXT NOT NULL, text TEXT NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}', hash TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
  embedded INTEGER NOT NULL DEFAULT 0, model_key TEXT);
"""


def default_path():
    root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("PRICES_DATA_DIR") or root / "Massive") / "ai_memory.sqlite3"


def memory_hash(kind, ticker, text, dedupe_key=None):
    key = str(dedupe_key) if dedupe_key not in (None, "") else re.sub(r"\s+", " ", str(text)).strip()
    return hashlib.sha256(f"{kind}|{ticker or ''}|{key}".encode("utf-8")).hexdigest()


def _iso(ts):
    if ts is None:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    value = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.astimezone()
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def ingest(kind, text, ticker=None, ts=None, meta=None, db_path=None):
    """Returns (id, duplicate). Never raises on duplicates; raises ValueError on bad input."""
    kind = str(kind).upper()
    if kind not in KINDS:
        raise ValueError(f"unknown memory kind {kind}")
    text = str(text or "").strip()
    if not text:
        raise ValueError("empty memory text")
    ticker = str(ticker).upper().strip() if ticker else None
    meta = meta or {}
    digest = memory_hash(kind, ticker, text, meta.get("dedupeKey"))
    path = Path(db_path or default_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        cur = db.execute("INSERT OR IGNORE INTO memory_items (kind, ticker, ts, text, meta, hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (kind, ticker, _iso(ts), text, json.dumps(meta), digest, _iso(None)))
        if cur.rowcount:
            return cur.lastrowid, False
        return db.execute("SELECT id FROM memory_items WHERE hash=?", (digest,)).fetchone()[0], True

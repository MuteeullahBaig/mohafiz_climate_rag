"""Free-tier survival: response cache + daily LLM budget (framework-agnostic).

A public demo on free-tier keys must assume abuse and repetition. Two layers here:
  1. Response cache — disaster-prep questions repeat heavily ("what to do in a flood"),
     so an exact+normalized cache serves them at zero LLM cost.
  2. Daily budget — an atomic per-day counter (Asia/Karachi) caps LLM answers; when
     exhausted the app degrades to retrieval-only instead of erroring.

Per-IP rate limiting lives in the FastAPI layer (slowapi), where the request/IP is.
SQLite state lives under data/ (ephemeral on HF Spaces restarts — fine for a demo).
"""
import hashlib
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone, timedelta

import config

DB = config.DATA / "quota.sqlite"
DAILY_LLM_BUDGET = int(os.environ.get("DAILY_LLM_BUDGET", "300"))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", str(7 * 24 * 3600)))
PK_TZ = timezone(timedelta(hours=5))  # Asia/Karachi

_lock = threading.Lock()
_conn = None


def _db():
    global _conn
    if _conn is None:
        config.DATA.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB), check_same_thread=False)
        _conn.execute("CREATE TABLE IF NOT EXISTS cache "
                      "(key TEXT PRIMARY KEY, value TEXT, ts REAL)")
        _conn.execute("CREATE TABLE IF NOT EXISTS budget (day TEXT PRIMARY KEY, used INTEGER)")
        _conn.commit()
    return _conn


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _key(q: str) -> str:
    return hashlib.sha256(_norm(q).encode("utf-8")).hexdigest()


def pk_today() -> str:
    return datetime.now(PK_TZ).strftime("%Y-%m-%d")


# ─── response cache ─────────────────────────────────────────────────────────────
def cache_get(query: str):
    import json
    with _lock:
        row = _db().execute("SELECT value, ts FROM cache WHERE key=?", (_key(query),)).fetchone()
    if not row:
        return None
    value, ts = row
    if time.time() - ts > CACHE_TTL_SECONDS:
        return None
    return json.loads(value)


def cache_put(query: str, value: dict):
    import json
    with _lock:
        _db().execute("INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?,?,?)",
                      (_key(query), json.dumps(value, ensure_ascii=False), time.time()))
        _db().commit()


# ─── daily LLM budget ───────────────────────────────────────────────────────────
def try_consume_budget(n: int = 1) -> bool:
    """Atomically reserve n LLM calls against today's cap. False if it would exceed."""
    day = pk_today()
    with _lock:
        cur = _db().execute("SELECT used FROM budget WHERE day=?", (day,)).fetchone()
        used = cur[0] if cur else 0
        if used + n > DAILY_LLM_BUDGET:
            return False
        _db().execute("INSERT OR REPLACE INTO budget (day, used) VALUES (?,?)", (day, used + n))
        _db().commit()
        return True


def budget_status() -> dict:
    day = pk_today()
    with _lock:
        cur = _db().execute("SELECT used FROM budget WHERE day=?", (day,)).fetchone()
    used = cur[0] if cur else 0
    return {"day": day, "used": used, "cap": DAILY_LLM_BUDGET,
            "remaining": max(0, DAILY_LLM_BUDGET - used)}


if __name__ == "__main__":
    print("budget:", budget_status())
    cache_put("test q", {"answer": "hi"})
    print("cache hit:", cache_get("Test  Q"))  # normalized match
    print("consume:", try_consume_budget(1), budget_status())

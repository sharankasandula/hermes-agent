"""Mnemosyne local SQLite memory provider for Hermes.

This provider intentionally starts simple and conservative:
- profile-scoped storage under ``$HERMES_HOME/mnemosyne/data/mnemosyne.db``
- no automatic LLM fact extraction
- short FTS recall blocks, capped to avoid prompt pollution
- explicit ``mnemosyne_remember`` / ``mnemosyne_recall`` / ``mnemosyne_stats`` tools
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_id(text: str, *, salt: str = "") -> str:
    return hashlib.sha256((salt + "\0" + text).encode("utf-8", "ignore")).hexdigest()[:16]


def _clean_text(value: Any, limit: int = 12000) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit].strip()


LOW_VALUE_TURN_RE = re.compile(
    r"(?i)^(done|fixed|shipped|all verified|fantastic|great|thanks|thank you|ok|okay|yes|no|cool|nice)[.!\s]*$"
)
LOG_DUMP_RE = re.compile(
    r"(?i)(traceback \(most recent call last\)|npm ERR!|gitleaks|secret-audit|/tmp/.*secret|BEGIN [A-Z ]*PRIVATE KEY|api[_-]?key|access[_-]?token|refresh[_-]?token|password\s*[:=]|-----BEGIN|```.{1500,}```)"
)
LOW_VALUE_PHRASE_RE = re.compile(
    r"(?i)(done\.?\s*$|shipped this turn|all verified, committed|log dump|full log|nothing else urgent|can we consider this done|cleanup complete)"
)
HIGH_SIGNAL_RE = re.compile(
    r"(?i)(hard rule|preference|prefers|canonical|source of truth|ssot|decision|decided|requirement|must|do not|never|blocked|blocker|root cause|migrat(?:ed|ion)|decommission|installed|configured|verified|public exposure|secret manager|bitwarden|german pension|a1310|versicherungsnummer|services\.yml|handoff|runbook|workflow|policy)"
)


def _iso_after(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).replace(microsecond=0).isoformat()


def _turn_quality(user: str, assistant: str) -> tuple[bool, float, Optional[str], str]:
    """Return (store, importance, valid_until, reason).

    Explicit memory writes bypass this; this only governs passive turn mirroring.
    """
    combined = f"{user}\n{assistant}"
    word_count = len(re.findall(r"\w+", combined))
    if not user or not assistant:
        return False, 0.0, None, "empty"
    if LOG_DUMP_RE.search(combined):
        return False, 0.0, None, "secret_or_log_dump"
    if LOW_VALUE_TURN_RE.match(user.strip()) and word_count < 80:
        return False, 0.0, None, "ack_only"
    if LOW_VALUE_PHRASE_RE.search(combined) and not HIGH_SIGNAL_RE.search(combined):
        return False, 0.0, None, "low_value_done_shipped"
    if HIGH_SIGNAL_RE.search(combined):
        # High-signal project/rule/decision memory. Keep longer and rank above routine turns.
        return True, 0.55, _iso_after(365), "high_signal"
    if word_count < 45:
        return False, 0.0, None, "too_short_routine"
    # Routine completed exchange: retain briefly at low importance so it can help
    # immediate follow-ups, then weekly maintenance prunes it.
    return True, 0.12, _iso_after(14), "routine_ttl"


class MnemosyneMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "mnemosyne"

    def __init__(self) -> None:
        self._hermes_home: Optional[Path] = None
        self._db_path: Optional[Path] = None
        self._session_id = ""
        self._profile = "default"
        self._max_prefetch_chars = 4000
        self._prefetch_limit = 5

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = Path(kwargs.get("hermes_home") or Path.home() / ".hermes").expanduser()
        self._hermes_home = hermes_home
        self._session_id = session_id or ""
        self._profile = str(kwargs.get("agent_identity") or "default")
        self._db_path = hermes_home / "mnemosyne" / "data" / "mnemosyne.db"
        try:
            from hermes_cli.config import load_config
            cfg = ((load_config() or {}).get("memory", {}) or {}).get("mnemosyne", {}) or {}
            self._prefetch_limit = max(1, min(int(cfg.get("recall_limit", self._prefetch_limit)), 20))
            self._max_prefetch_chars = max(500, min(int(cfg.get("max_prefetch_chars", self._max_prefetch_chars)), 12000))
        except Exception:
            pass
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def system_prompt_block(self) -> str:
        return (
            "Mnemosyne memory is active. Treat recalled memory as helpful but dated; "
            "prefer live system checks and current profile MEMORY.md/USER.md when they conflict."
        )

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # Store only compact completed-turn summaries. Tool outputs/raw transcripts
        # are intentionally excluded to avoid re-ingesting secrets or stale dumps.
        user = _clean_text(user_content, 3000)
        assistant = _clean_text(assistant_content, 3000)
        store, importance, valid_until, reason = _turn_quality(user, assistant)
        if not store:
            return
        content = f"[USER] {user}\n[ASSISTANT] {assistant}"
        self._remember(
            content,
            source="conversation",
            session_id=session_id or self._session_id,
            importance=importance,
            metadata={"profile": self._profile, "kind": "turn", "quality": reason},
            memory_type="turn",
            valid_until=valid_until,
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        query = _clean_text(query, 500)
        if not query:
            return ""
        rows = self._recall(query, limit=self._prefetch_limit)
        if not rows:
            return ""
        parts: List[str] = []
        total = 0
        for idx, row in enumerate(rows, start=1):
            text = row["content"].strip()
            source = row.get("source") or "memory"
            ts = row.get("timestamp") or ""
            chunk = f"[{idx}] source={source} time={ts}\n{text}"
            if total + len(chunk) > self._max_prefetch_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        text = _clean_text(content, 6000)
        if not text:
            return
        if action == "remove":
            # Soft-delete exact/near-exact current built-in memory mirrors.
            self._soft_delete(text, source_prefix=f"builtin_{target}")
            return
        self._remember(
            text,
            source=f"builtin_{target}",
            session_id=(metadata or {}).get("session_id") or self._session_id,
            importance=0.85 if target == "user" else 0.75,
            metadata={"target": target, "action": action, **(metadata or {})},
            memory_type=target,
            veracity="stated",
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "mnemosyne_remember",
                "description": "Store a durable memory in local Mnemosyne. Use only for stable facts/preferences, not task progress.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content to store."},
                        "source": {"type": "string", "description": "Optional source label.", "default": "manual"},
                        "importance": {"type": "number", "description": "0.0-1.0 importance score.", "default": 0.6},
                        "metadata": {"type": "object", "description": "Optional JSON metadata."},
                        "veracity": {"type": "string", "description": "unknown/stated/verified/superseded.", "default": "stated"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "mnemosyne_recall",
                "description": "Search local Mnemosyne memory for relevant past context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "limit": {"type": "integer", "description": "Max results, default 5.", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "mnemosyne_stats",
                "description": "Return local Mnemosyne database statistics.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "mnemosyne_remember":
            content = _clean_text(args.get("content"), 12000)
            if not content:
                return json.dumps({"status": "error", "error": "content is required"})
            memory_id = self._remember(
                content,
                source=str(args.get("source") or "manual"),
                session_id=kwargs.get("session_id") or self._session_id,
                importance=float(args.get("importance", 0.6) or 0.6),
                metadata=args.get("metadata") if isinstance(args.get("metadata"), dict) else None,
                memory_type="manual",
                veracity=str(args.get("veracity") or "stated"),
            )
            return json.dumps({"status": "stored", "memory_id": memory_id, "content_preview": content[:120]})
        if tool_name == "mnemosyne_recall":
            query = _clean_text(args.get("query"), 500)
            limit = max(1, min(int(args.get("limit", 5) or 5), 20))
            rows = self._recall(query, limit=limit)
            return json.dumps({"query": query, "count": len(rows), "results": rows}, ensure_ascii=False)
        if tool_name == "mnemosyne_stats":
            return json.dumps(self._stats(), ensure_ascii=False)
        raise NotImplementedError(tool_name)

    def _connect(self) -> sqlite3.Connection:
        if not self._db_path:
            raise RuntimeError("Mnemosyne provider not initialized")
        con = sqlite3.connect(str(self._db_path), timeout=15)
        con.row_factory = sqlite3.Row
        return con

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    importance REAL DEFAULT 0.5,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    profile TEXT,
                    scope TEXT DEFAULT 'profile',
                    memory_type TEXT DEFAULT 'memory',
                    veracity TEXT DEFAULT 'unknown',
                    valid_until TEXT,
                    compacted_at TEXT,
                    deleted INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS working_memory (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    importance REAL DEFAULT 0.5,
                    metadata_json TEXT DEFAULT '{}',
                    veracity TEXT DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    memory_type TEXT DEFAULT 'memory',
                    scope TEXT DEFAULT 'profile',
                    profile TEXT,
                    valid_until TEXT,
                    compacted_at TEXT,
                    deleted INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    timestamp TEXT,
                    source_msg_id TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
                    content, source, metadata_json, content='memories', content_rowid='rowid'
                );
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO fts_memories(rowid, content, source, metadata_json)
                    VALUES (new.rowid, new.content, coalesce(new.source,''), coalesce(new.metadata_json,''));
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO fts_memories(fts_memories, rowid, content, source, metadata_json)
                    VALUES ('delete', old.rowid, old.content, coalesce(old.source,''), coalesce(old.metadata_json,''));
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO fts_memories(fts_memories, rowid, content, source, metadata_json)
                    VALUES ('delete', old.rowid, old.content, coalesce(old.source,''), coalesce(old.metadata_json,''));
                    INSERT INTO fts_memories(rowid, content, source, metadata_json)
                    VALUES (new.rowid, new.content, coalesce(new.source,''), coalesce(new.metadata_json,''));
                END;
                """
            )
            for table in ("memories", "working_memory"):
                cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
                for col in ("valid_until", "compacted_at"):
                    if col not in cols:
                        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")

    def _remember(
        self,
        content: str,
        *,
        source: str,
        session_id: str = "",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: str = "memory",
        veracity: str = "unknown",
        valid_until: Optional[str] = None,
    ) -> str:
        self._ensure_schema()
        ts = _now()
        metadata = metadata or {}
        metadata.setdefault("profile", self._profile)
        mid = _stable_id(content, salt=f"{source}:{session_id}:{memory_type}")
        mjson = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        row = (mid, content, source, ts, session_id, float(importance), mjson, self._profile, memory_type, veracity, valid_until)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO memories(id, content, source, timestamp, session_id, importance, metadata_json, profile, memory_type, veracity, valid_until, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    source=excluded.source,
                    timestamp=excluded.timestamp,
                    session_id=excluded.session_id,
                    importance=excluded.importance,
                    metadata_json=excluded.metadata_json,
                    profile=excluded.profile,
                    memory_type=excluded.memory_type,
                    veracity=excluded.veracity,
                    valid_until=excluded.valid_until,
                    deleted=0
                """,
                row,
            )
            con.execute(
                """
                INSERT INTO working_memory(id, content, source, timestamp, session_id, importance, metadata_json, profile, memory_type, veracity, valid_until, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    source=excluded.source,
                    timestamp=excluded.timestamp,
                    session_id=excluded.session_id,
                    importance=excluded.importance,
                    metadata_json=excluded.metadata_json,
                    profile=excluded.profile,
                    memory_type=excluded.memory_type,
                    veracity=excluded.veracity,
                    valid_until=excluded.valid_until,
                    deleted=0
                """,
                row,
            )
        return mid

    def _soft_delete(self, content: str, *, source_prefix: str) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE memories SET deleted=1 WHERE source LIKE ? AND content=?",
                (f"{source_prefix}%", content),
            )
            con.execute(
                "UPDATE working_memory SET deleted=1 WHERE source LIKE ? AND content=?",
                (f"{source_prefix}%", content),
            )

    def _recall(self, query: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []
        self._ensure_schema()
        with self._connect() as con:
            try:
                rows = con.execute(
                    """
                    SELECT m.id, m.content, m.source, m.timestamp, m.session_id, m.importance,
                           m.metadata_json, bm25(fts_memories) AS rank
                    FROM fts_memories
                    JOIN memories m ON m.rowid = fts_memories.rowid
                    WHERE fts_memories MATCH ? AND coalesce(m.deleted,0)=0
                      AND (m.valid_until IS NULL OR m.valid_until='' OR m.valid_until > datetime('now'))
                    ORDER BY rank ASC, m.importance DESC, m.timestamp DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), limit),
                ).fetchall()
            except sqlite3.Error:
                like = f"%{query}%"
                rows = con.execute(
                    """
                    SELECT id, content, source, timestamp, session_id, importance, metadata_json, 0 AS rank
                    FROM memories
                    WHERE coalesce(deleted,0)=0 AND content LIKE ?
                      AND (valid_until IS NULL OR valid_until='' OR valid_until > datetime('now'))
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT ?
                    """,
                    (like, limit),
                ).fetchall()
        return [self._row_to_result(r) for r in rows]

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = re.findall(r"[\w./:@-]+", query, flags=re.UNICODE)
        terms = [t for t in terms if len(t) >= 2][:12]
        if not terms:
            return '""'
        return " OR ".join(f'"{t}"' for t in terms)

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> Dict[str, Any]:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except Exception:
            metadata = {}
        return {
            "id": row["id"],
            "content": row["content"],
            "source": row["source"],
            "timestamp": row["timestamp"],
            "session_id": row["session_id"],
            "importance": row["importance"],
            "metadata": metadata,
        }

    def _stats(self) -> Dict[str, Any]:
        self._ensure_schema()
        with self._connect() as con:
            def count(table: str) -> int:
                return int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            by_source = [
                {"source": r[0], "count": r[1]}
                for r in con.execute(
                    "SELECT source, count(*) FROM memories WHERE coalesce(deleted,0)=0 GROUP BY source ORDER BY count(*) DESC LIMIT 20"
                ).fetchall()
            ]
            return {
                "provider": "mnemosyne",
                "db_path": str(self._db_path),
                "profile": self._profile,
                "memories": count("memories"),
                "working_memory": count("working_memory"),
                "facts": count("facts"),
                "by_source": by_source,
            }


def register(ctx) -> None:
    ctx.register_memory_provider(MnemosyneMemoryProvider())

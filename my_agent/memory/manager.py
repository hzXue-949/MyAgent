"""第 8.1–8.2 节：本地优先的分层记忆系统与 MemoryTool。"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import uuid
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from hello_agents.tools import Tool, ToolParameter

from my_agent.paths import DATA_DIR, ENV_FILE, PROJECT_ROOT


MemoryType = Literal["working", "episodic", "semantic", "perceptual"]
MEMORY_TYPES = {"working", "episodic", "semantic", "perceptual"}


def _load_dotenv() -> None:
    """加载项目 .env；缺少 python-dotenv 时提供小型兼容读取器。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE)
    except ImportError:
        path = ENV_FILE
        if not path.is_file():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            os.environ.setdefault(key.strip(), value)


@dataclass
class MemoryConfig:
    """记忆生命周期及本地持久化配置。"""

    database_path: str = str(DATA_DIR / "memory.sqlite3")
    working_memory_capacity: int = 50
    working_memory_ttl_minutes: int = 60
    default_search_limit: int = 5
    min_importance: float = 0.0


@dataclass
class MemoryItem:
    """统一的记忆条目结构，方便不同记忆类型共享管理接口。"""

    id: str
    user_id: str
    memory_type: MemoryType
    content: str
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["created_at"] = _iso_time(self.created_at)
        item["updated_at"] = _iso_time(self.updated_at)
        return item


def _iso_time(value: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _tokens(text: str) -> list[str]:
    """按英文词和中文字符二元组构造轻量检索 token。"""
    tokens: list[str] = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9_]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def _lexical_similarity(query: str, content: str) -> float:
    q, d = Counter(_tokens(query)), Counter(_tokens(content))
    if not q or not d:
        return 0.0
    dot = sum(count * d[token] for token, count in q.items())
    q_norm = math.sqrt(sum(value * value for value in q.values()))
    d_norm = math.sqrt(sum(value * value for value in d.values()))
    cosine = dot / (q_norm * d_norm) if q_norm and d_norm else 0.0
    phrase = 1.0 if query.strip().lower() in content.lower() else 0.0
    return min(1.0, cosine * 0.85 + phrase * 0.15)


class SQLiteMemoryStore:
    """SQLite 持久化条目仓库；工作记忆由 WorkingMemory 单独保存在内存中。"""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        if database_path != ":memory:":
            path = Path(database_path).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            self.database_path = str(path)
        self._memory_connection = sqlite3.connect(":memory:") if database_path == ":memory:" else None
        if self._memory_connection is not None:
            self._memory_connection.row_factory = sqlite3.Row
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = self._memory_connection if self._memory_connection is not None else sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            if self._memory_connection is None:
                connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    session_id TEXT,
                    metadata TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0
                )"""
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_type_time "
                "ON memories(user_id, memory_type, created_at)"
            )

    def save(self, item: MemoryItem) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content, importance=excluded.importance,
                    updated_at=excluded.updated_at, session_id=excluded.session_id,
                    metadata=excluded.metadata, access_count=excluded.access_count""",
                (
                    item.id, item.user_id, item.memory_type, item.content,
                    item.importance, item.created_at, item.updated_at, item.session_id,
                    json.dumps(item.metadata, ensure_ascii=False), item.access_count,
                ),
            )

    def get(self, user_id: str, memory_id: str) -> MemoryItem | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM memories WHERE user_id=? AND id=?", (user_id, memory_id)
            ).fetchone()
        return _row_to_item(row) if row else None

    def list(self, user_id: str, memory_types: set[str] | None = None) -> list[MemoryItem]:
        if memory_types is not None and not memory_types:
            return []
        sql, args = "SELECT * FROM memories WHERE user_id=?", [user_id]
        if memory_types:
            marks = ",".join("?" for _ in memory_types)
            sql += f" AND memory_type IN ({marks})"
            args.extend(sorted(memory_types))
        sql += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(sql, args).fetchall()
        return [_row_to_item(row) for row in rows]

    def delete(self, user_id: str, memory_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                "DELETE FROM memories WHERE user_id=? AND id=?", (user_id, memory_id)
            )
            return cursor.rowcount > 0

    def delete_many(self, user_id: str, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        marks = ",".join("?" for _ in memory_ids)
        with self._connect() as db:
            cursor = db.execute(
                f"DELETE FROM memories WHERE user_id=? AND id IN ({marks})",
                [user_id, *memory_ids],
            )
            return cursor.rowcount

    def clear_user(self, user_id: str) -> int:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
            return cursor.rowcount


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=row["id"], user_id=row["user_id"], memory_type=row["memory_type"],
        content=row["content"], importance=row["importance"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        session_id=row["session_id"], metadata=json.loads(row["metadata"]),
        access_count=row["access_count"],
    )


class WorkingMemory:
    """会话内临时记忆：纯内存、TTL 清理，并在容量满时淘汰低优先级条目。"""

    def __init__(self, capacity: int, ttl_minutes: int) -> None:
        self.capacity = capacity
        self.ttl_seconds = ttl_minutes * 60
        self._items: dict[str, MemoryItem] = {}

    def _expire(self) -> None:
        now = time.time()
        expired = [key for key, item in self._items.items() if now - item.created_at > self.ttl_seconds]
        for key in expired:
            del self._items[key]

    def add(self, item: MemoryItem) -> None:
        self._expire()
        if len(self._items) >= self.capacity:
            victim = min(self._items.values(), key=lambda m: (m.importance, m.created_at))
            del self._items[victim.id]
        self._items[item.id] = item

    def list(self, user_id: str) -> list[MemoryItem]:
        self._expire()
        return [item for item in self._items.values() if item.user_id == user_id]

    def delete(self, user_id: str, memory_id: str) -> bool:
        item = self._items.get(memory_id)
        if not item or item.user_id != user_id:
            return False
        del self._items[memory_id]
        return True

    def clear_user(self, user_id: str) -> int:
        targets = [key for key, item in self._items.items() if item.user_id == user_id]
        for key in targets:
            del self._items[key]
        return len(targets)


class MemoryManager:
    """统一管理四种记忆类型，并隔离不同 user_id 的读写。"""

    def __init__(
        self, user_id: str = "default_user", config: MemoryConfig | None = None,
        enable_perceptual: bool = True,
    ) -> None:
        _load_dotenv()
        if not user_id.strip():
            raise ValueError("user_id 不能为空")
        self.user_id = user_id
        self.config = config or MemoryConfig(
            database_path=os.getenv("MEMORY_DB_PATH", str(DATA_DIR / "memory.sqlite3")),
            working_memory_capacity=int(os.getenv("WORKING_MEMORY_CAPACITY", "50")),
            working_memory_ttl_minutes=int(os.getenv("WORKING_MEMORY_TTL_MINUTES", "60")),
        )
        self.store = SQLiteMemoryStore(self.config.database_path)
        self.working = WorkingMemory(
            self.config.working_memory_capacity,
            self.config.working_memory_ttl_minutes,
        )
        self.enabled_types = MEMORY_TYPES if enable_perceptual else MEMORY_TYPES - {"perceptual"}
        from my_agent.memory.backends import SQLiteVectorStore
        from my_agent.memory.embedding import get_text_embedder

        self.embedder = get_text_embedder()
        # Qdrant 不可用时，仍使用同一个真实模型在 SQLite 中检索向量。
        self.local_vector_store = SQLiteVectorStore(self.store, self.embedder)
        self.vector_store = self.local_vector_store
        self.graph_store = None
        if os.getenv("QDRANT_URL"):
            try:
                from my_agent.memory.backends import QdrantVectorStore

                self.vector_store = QdrantVectorStore(embedder=self.embedder)
            except Exception as exc:
                warnings.warn(
                    f"Qdrant 未启用，将使用相同 Embedding 模型的 SQLite 向量索引（{type(exc).__name__}）。",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if os.getenv("NEO4J_URI") and os.getenv("NEO4J_PASSWORD"):
            try:
                from my_agent.memory.backends import Neo4jGraphStore

                self.graph_store = Neo4jGraphStore()
            except Exception as exc:
                warnings.warn(
                    f"Neo4j 未启用，改用本地记忆检索（{type(exc).__name__}）。",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _upsert_indexes(self, item: MemoryItem) -> None:
        from my_agent.memory.embedding import EmbeddingError

        if item.memory_type in {"semantic", "episodic"}:
            try:
                self.vector_store.upsert(item)
            except EmbeddingError as exc:
                raise RuntimeError(f"原文已保存（id={item.id}），向量未完成：{exc}") from exc
            except Exception as exc:
                # 保存原文后报告索引失败；下次检索会按内容哈希重试补建。
                raise RuntimeError(
                    f"记忆原文已保存（id={item.id}），但向量索引失败；请检查嵌入模型/网络后重新检索"
                ) from exc
        for name, backend in (("Neo4j", self.graph_store if item.memory_type == "semantic" else None),):
            if backend:
                try:
                    backend.upsert(item)
                except Exception as exc:
                    warnings.warn(
                        f"{name} 同步失败，本地 SQLite 条目已保存（{type(exc).__name__}）。",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    def _delete_indexes(self, item: MemoryItem) -> None:
        operations = []
        if item.memory_type in {"semantic", "episodic"}:
            operations.append(("向量索引", lambda: self.vector_store.delete(item.id)))
            if self.vector_store is not self.local_vector_store:
                operations.append(("SQLite 向量索引", lambda: self.local_vector_store.delete(item.id)))
        if self.graph_store and item.memory_type == "semantic":
            operations.append(("Neo4j", lambda: self.graph_store.delete_memory_links(item.id, self.user_id)))
        for name, operation in operations:
            try:
                operation()
            except Exception as exc:
                warnings.warn(
                    f"{name} 索引清理失败；SQLite 主记录已处理（{type(exc).__name__}）。",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def add_memory(
        self, content: str, memory_type: MemoryType = "working", importance: float = 0.5,
        metadata: dict[str, Any] | None = None, session_id: str | None = None,
    ) -> str:
        if memory_type not in self.enabled_types:
            raise ValueError(f"未启用或不支持的记忆类型: {memory_type}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 不能为空")
        if not 0 <= importance <= 1:
            raise ValueError("importance 必须在 0 到 1 之间")
        now = time.time()
        item = MemoryItem(
            id=str(uuid.uuid4()), user_id=self.user_id, memory_type=memory_type,
            content=content.strip(), importance=float(importance), created_at=now,
            updated_at=now, session_id=session_id,
            metadata=dict(metadata or {}),
        )
        if memory_type == "working":
            self.working.add(item)
        else:
            self.store.save(item)
        self._upsert_indexes(item)
        return item.id

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        item = self.store.get(self.user_id, memory_id)
        if item:
            return item
        for item in self.working.list(self.user_id):
            if item.id == memory_id:
                return item
        return None

    def retrieve_memories(
        self, query: str, limit: int = 5, memory_types: list[str] | None = None,
        min_importance: float = 0.0, session_id: str | None = None,
    ) -> list[MemoryItem]:
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        if not query.strip():
            raise ValueError("query 不能为空")
        requested = set(memory_types or self.enabled_types)
        invalid = requested - self.enabled_types
        if invalid:
            raise ValueError(f"不支持或未启用的记忆类型: {', '.join(sorted(invalid))}")
        items = self.store.list(self.user_id, requested - {"working"})
        if "working" in requested:
            items.extend(self.working.list(self.user_id))
        now = time.time()
        vector_scores: dict[str, float] = {}
        graph_scores: dict[str, float] = {}
        indexed_types = requested & {"semantic", "episodic"}
        index_candidates = [item for item in items if item.memory_type in indexed_types
                            and item.importance >= min_importance and (not session_id or item.session_id == session_id)]
        if index_candidates:
            from my_agent.memory.embedding import EmbeddingError

            try:
                self.vector_store.ensure_indexed(index_candidates)
                vector_scores = self.vector_store.search(
                    query, self.user_id, max(limit * 5, 10), indexed_types, session_id, min_importance,
                )
            except EmbeddingError:
                raise
            except Exception as exc:
                if self.vector_store is self.local_vector_store:
                    raise RuntimeError("Embedding 生成或本地向量检索失败，请检查模型配置与依赖") from exc
                warnings.warn(f"Qdrant 检索失败，改用同模型的本地向量检索（{type(exc).__name__}）。", RuntimeWarning, stacklevel=2)
                vector_scores = self.local_vector_store.search(
                    query, self.user_id, max(limit * 5, 10), indexed_types, session_id, min_importance,
                )
        if "semantic" in requested and self.graph_store:
            try:
                graph_scores = self.graph_store.search(query, self.user_id, max(limit * 3, 10))
            except Exception as exc:
                warnings.warn(f"Neo4j 检索失败，使用现有向量/词项结果（{type(exc).__name__}）。", RuntimeWarning, stacklevel=2)
        scored: list[tuple[float, MemoryItem]] = []
        for item in items:
            if item.importance < min_importance:
                continue
            if session_id and item.session_id != session_id:
                continue
            similarity = _lexical_similarity(query, item.content)
            recency = math.exp(-max(0, now - item.created_at) / (30 * 86400))
            if item.memory_type == "episodic":
                if item.id not in vector_scores:
                    continue
                score = (vector_scores[item.id] * 0.8 + recency * 0.2) * (0.8 + item.importance * 0.4)
            elif item.memory_type == "semantic":
                if item.id not in vector_scores and item.id not in graph_scores:
                    continue
                vector_score = vector_scores.get(item.id, 0.0)
                graph_score = graph_scores.get(item.id, 0.0)
                score = (vector_score * 0.7 + graph_score * 0.3) * (0.8 + item.importance * 0.4)
            else:
                score = (similarity * 0.8 + recency * 0.2) * (0.8 + item.importance * 0.4)
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        results = [item for _, item in scored[:limit]]
        # 记录访问次数，供统计与遗忘策略使用。
        for item in results:
            item.access_count += 1
            item.updated_at = now
            if item.memory_type != "working":
                self.store.save(item)
        return results

    def reindex_memories(self, force: bool = False) -> int:
        """用当前模型补建当前用户的情景/语义向量；保留原文和旧模型集合。"""
        items = self.store.list(self.user_id, {"semantic", "episodic"})
        return self.vector_store.ensure_indexed(items, force=force)

    def update_memory(
        self, memory_id: str, content: str | None = None,
        importance: float | None = None, metadata: dict[str, Any] | None = None,
    ) -> bool:
        item = self.get_memory(memory_id)
        if not item:
            return False
        if content is not None:
            if not content.strip():
                raise ValueError("content 不能为空")
            item.content = content.strip()
        if importance is not None:
            if not 0 <= importance <= 1:
                raise ValueError("importance 必须在 0 到 1 之间")
            item.importance = float(importance)
        if metadata is not None:
            item.metadata.update(metadata)
        item.updated_at = time.time()
        if item.memory_type == "working":
            self.working._items[item.id] = item
        else:
            self.store.save(item)
        self._upsert_indexes(item)
        return True

    def remove_memory(self, memory_id: str) -> bool:
        if self.working.delete(self.user_id, memory_id):
            return True
        item = self.store.get(self.user_id, memory_id)
        if not item:
            return False
        deleted = self.store.delete(self.user_id, memory_id)
        if deleted:
            self._delete_indexes(item)
        return deleted

    def forget_memories(
        self, strategy: str = "importance_based", threshold: float = 0.1,
        max_age_days: int = 30, max_items: int | None = None,
    ) -> int:
        items = self.store.list(self.user_id)
        now = time.time()
        if strategy == "importance_based":
            targets = [item for item in items if item.importance < threshold]
        elif strategy == "time_based":
            targets = [item for item in items if now - item.updated_at > max_age_days * 86400]
        elif strategy == "capacity_based":
            if max_items is None or max_items < 0:
                raise ValueError("capacity_based 策略需要非负 max_items")
            overflow = max(0, len(items) - max_items)
            targets = sorted(items, key=lambda item: (item.importance, item.access_count, item.updated_at))[:overflow]
        else:
            raise ValueError("strategy 支持 importance_based、time_based、capacity_based")
        persistent_count = self.store.delete_many(self.user_id, [item.id for item in targets])
        for item in targets:
            self._delete_indexes(item)
        if strategy == "capacity_based" and max_items is not None:
            working = self.working.list(self.user_id)
            overflow = max(0, len(working) - max_items)
            victims = sorted(working, key=lambda item: (item.importance, item.created_at))[:overflow]
            persistent_count += sum(self.working.delete(self.user_id, item.id) for item in victims)
        else:
            for item in self.working.list(self.user_id):
                expired = strategy == "time_based" and now - item.updated_at > max_age_days * 86400
                low_importance = strategy == "importance_based" and item.importance < threshold
                if expired or low_importance:
                    persistent_count += int(self.working.delete(self.user_id, item.id))
        return persistent_count

    def consolidate(self, session_id: str | None = None, min_importance: float = 0.5) -> int:
        candidates = [
            item for item in self.working.list(self.user_id)
            if item.importance >= min_importance and (not session_id or item.session_id == session_id)
        ]
        moved = 0
        for item in candidates:
            # 情景记忆保留具体事件，语义记忆仍由调用者明确选择，避免错误抽象。
            self.add_memory(item.content, "episodic", item.importance,
                            {**item.metadata, "consolidated_from": item.id}, item.session_id)
            self.working.delete(self.user_id, item.id)
            moved += 1
        return moved

    def clear_all(self) -> int:
        items = self.store.list(self.user_id)
        count = self.store.clear_user(self.user_id) + self.working.clear_user(self.user_id)
        for item in items:
            self._delete_indexes(item)
        return count

    def stats(self) -> dict[str, Any]:
        items = self.store.list(self.user_id) + self.working.list(self.user_id)
        counts = Counter(item.memory_type for item in items)
        return {
            "user_id": self.user_id,
            "total": len(items),
            "by_type": {kind: counts.get(kind, 0) for kind in sorted(self.enabled_types)},
            "average_importance": round(sum(item.importance for item in items) / len(items), 3) if items else 0.0,
            "embedding_provider": os.getenv("EMBED_MODEL_TYPE") or "local",
            "embedding_model": self.embedder.model_name,
            "vector_backend": type(self.vector_store).__name__,
        }

    def summary(self, limit: int = 10) -> dict[str, Any]:
        items = self.store.list(self.user_id) + self.working.list(self.user_id)
        items.sort(key=lambda item: (item.importance, item.updated_at), reverse=True)
        return {**self.stats(), "recent_important": [item.to_dict() for item in items[:limit]]}


class MemoryTool(Tool):
    """Hello-Agents 工具注册表可直接使用的记忆工具。"""

    def __init__(
        self, user_id: str = "default_user", memory_config: MemoryConfig | None = None,
        enable_perceptual: bool = True,
    ) -> None:
        super().__init__(name="memory", description="添加、检索和管理用户的工作、情景、语义及感知记忆")
        self.manager = MemoryManager(user_id, memory_config, enable_perceptual)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="action", type="string", description="add/search/summary/stats/update/remove/forget/consolidate/clear_all", required=True),
            ToolParameter(name="content", type="string", description="记忆文本；add/update 时使用", required=False),
            ToolParameter(name="memory_type", type="string", description="working/episodic/semantic/perceptual", required=False),
            ToolParameter(name="query", type="string", description="检索内容", required=False),
            ToolParameter(name="memory_types", type="array", description="搜索范围，例如 [episodic, semantic]", required=False),
            ToolParameter(name="memory_id", type="string", description="记忆 ID", required=False),
            ToolParameter(name="session_id", type="string", description="会话 ID", required=False),
            ToolParameter(name="importance", type="number", description="0 到 1 的重要性", required=False),
            ToolParameter(name="limit", type="integer", description="返回条数", required=False),
            ToolParameter(name="metadata", type="object", description="结构化元数据；可包含 entities/relations", required=False),
            ToolParameter(name="min_importance", type="number", description="检索或整合的最低重要性", required=False),
            ToolParameter(name="strategy", type="string", description="importance_based/time_based/capacity_based", required=False),
            ToolParameter(name="threshold", type="number", description="重要性遗忘阈值", required=False),
            ToolParameter(name="max_age_days", type="integer", description="时间遗忘天数", required=False),
            ToolParameter(name="max_items", type="integer", description="容量遗忘策略保留条数", required=False),
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        """按 Hello-Agents Tool 的标准接口接收参数字典。"""
        action = parameters.get("action", "")
        kwargs = parameters
        try:
            if action == "add":
                memory_id = self.manager.add_memory(
                    kwargs.get("content", ""), kwargs.get("memory_type", "working"),
                    kwargs.get("importance", 0.5), kwargs.get("metadata"),
                    kwargs.get("session_id"),
                )
                return f"记忆已添加，id={memory_id}"
            if action == "search":
                results = self.manager.retrieve_memories(
                    kwargs.get("query", ""), kwargs.get("limit", self.manager.config.default_search_limit),
                    kwargs.get("memory_types"), kwargs.get("min_importance", self.manager.config.min_importance),
                    kwargs.get("session_id"),
                )
                return json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2)
            if action == "summary":
                return json.dumps(self.manager.summary(kwargs.get("limit", 10)), ensure_ascii=False, indent=2)
            if action == "stats":
                return json.dumps(self.manager.stats(), ensure_ascii=False, indent=2)
            if action == "update":
                changed = self.manager.update_memory(kwargs["memory_id"], kwargs.get("content"), kwargs.get("importance"), kwargs.get("metadata"))
                return "记忆已更新" if changed else "未找到该记忆"
            if action == "remove":
                return "记忆已删除" if self.manager.remove_memory(kwargs["memory_id"]) else "未找到该记忆"
            if action == "forget":
                count = self.manager.forget_memories(
                    kwargs.get("strategy", "importance_based"), kwargs.get("threshold", 0.1),
                    kwargs.get("max_age_days", 30), kwargs.get("max_items"),
                )
                return f"已遗忘 {count} 条记忆"
            if action == "consolidate":
                count = self.manager.consolidate(kwargs.get("session_id"), kwargs.get("min_importance", 0.5))
                return f"已整合 {count} 条工作记忆"
            if action == "clear_all":
                return f"已清空 {self.manager.clear_all()} 条记忆"
            raise ValueError(f"不支持的 action: {action}")
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return f"记忆操作失败：{exc}"

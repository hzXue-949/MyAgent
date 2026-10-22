"""第 8.2 节可选云端适配：Qdrant 向量索引与 Neo4j 关系图。"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .embedding import TextEmbedder, get_text_embedder


INDEXED_MEMORY_TYPES = {"semantic", "episodic"}


def _content_hash(item: Any) -> str:
    return hashlib.sha256(item.content.encode()).hexdigest()


class SQLiteVectorStore:
    """小规模本地向量索引：缓存真实模型向量，精确计算归一化点积。"""

    def __init__(self, document_store: Any, embedder: TextEmbedder) -> None:
        self.store = document_store
        self.embedder = embedder
        with self.store._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_vectors ("
                "memory_id TEXT NOT NULL, signature TEXT NOT NULL, "
                "content_hash TEXT NOT NULL, vector TEXT NOT NULL, "
                "PRIMARY KEY(memory_id, signature))"
            )

    def ensure_indexed(self, items: list[Any], force: bool = False) -> int:
        if not items:
            return 0
        signature = self.embedder.signature
        with self.store._connect() as db:
            known = {row["memory_id"]: row["content_hash"] for row in db.execute(
                "SELECT memory_id, content_hash FROM memory_vectors WHERE signature=?", (signature,)
            )}
        pending = [item for item in items if force or known.get(item.id) != _content_hash(item)]
        # 每个批次完成即持久化，失败后下次检索可继续补建。
        for offset in range(0, len(pending), 32):
            batch = pending[offset:offset + 32]
            vectors = self.embedder.encode_batch([item.content for item in batch])
            with self.store._connect() as db:
                db.executemany(
                    "INSERT OR REPLACE INTO memory_vectors VALUES (?, ?, ?, ?)",
                    [(item.id, signature, _content_hash(item), json.dumps(vector))
                     for item, vector in zip(batch, vectors)],
                )
        return len(pending)

    def upsert(self, item: Any) -> None:
        self.ensure_indexed([item])

    def search(self, query: str, user_id: str, limit: int = 10,
               memory_types: set[str] | None = None, session_id: str | None = None,
               min_importance: float = 0.0) -> dict[str, float]:
        candidates = [item for item in self.store.list(user_id, memory_types or INDEXED_MEMORY_TYPES)
                      if item.importance >= min_importance and (not session_id or item.session_id == session_id)]
        if not candidates:
            return {}
        self.ensure_indexed(candidates)
        query_vector = self.embedder.encode_query(query)
        with self.store._connect() as db:
            rows = db.execute(
                "SELECT v.memory_id, v.vector FROM memory_vectors v "
                "JOIN memories m ON m.id=v.memory_id WHERE m.user_id=? AND v.signature=?",
                (user_id, self.embedder.signature),
            ).fetchall()
        allowed = {item.id for item in candidates}
        scores = {row["memory_id"]: sum(a * b for a, b in zip(query_vector, json.loads(row["vector"])))
                  for row in rows if row["memory_id"] in allowed}
        return dict(sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit])

    def delete(self, memory_id: str) -> None:
        with self.store._connect() as db:
            db.execute("DELETE FROM memory_vectors WHERE memory_id=?", (memory_id,))


class QdrantVectorStore:
    """情景/语义向量索引；按嵌入模型身份隔离集合，SQLite 保留原文。"""

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 collection: str | None = None, embedder: TextEmbedder | None = None) -> None:
        from qdrant_client import QdrantClient, models

        self.models = models
        self.embedder = embedder or get_text_embedder()
        base_collection = collection or os.getenv("QDRANT_COLLECTION", "myagent_memory")
        # 同维度的不同模型也必须隔离；保留旧集合，不覆盖已有哈希向量。
        self.collection = f"{base_collection}_{self.embedder.signature}"
        self.client = QdrantClient(
            url=url or os.getenv("QDRANT_URL"),
            api_key=api_key or os.getenv("QDRANT_API_KEY"),
        )
        existing = {item.name for item in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.embedder.dimensions, distance=models.Distance.COSINE),
            )
        info = self.client.get_collection(self.collection)
        vectors_config = info.config.params.vectors
        if not isinstance(vectors_config, models.VectorParams) or vectors_config.size != self.embedder.dimensions or vectors_config.distance != models.Distance.COSINE:
            raise ValueError("Qdrant 集合的维度或距离配置与当前 Embedding 不一致")
        for field in ("user_id", "memory_type", "session_id"):
            if field not in info.payload_schema:
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD, wait=True)
        if "importance" not in info.payload_schema:
            self.client.create_payload_index(self.collection, "importance", models.PayloadSchemaType.FLOAT, wait=True)

    def upsert(self, item: Any) -> None:
        self.ensure_indexed([item], force=True)

    def ensure_indexed(self, items: list[Any], force: bool = False) -> int:
        from qdrant_client.models import PointStruct

        indexed = 0
        for offset in range(0, len(items), 32):
            batch = items[offset:offset + 32]
            known = {} if force else {
                str(point.id): point.payload or {} for point in self.client.retrieve(
                    self.collection, ids=[item.id for item in batch], with_payload=True, with_vectors=False,
                )
            }
            pending = [item for item in batch if force or known.get(item.id, {}).get("content_hash") != _content_hash(item)
                       or known.get(item.id, {}).get("importance") != item.importance
                       or known.get(item.id, {}).get("session_id") != item.session_id]
            if not pending:
                continue
            vectors = self.embedder.encode_batch([item.content for item in pending])
            self.client.upsert(
                collection_name=self.collection, wait=True,
                points=[PointStruct(id=item.id, vector=vector, payload={
                    "user_id": item.user_id, "memory_id": item.id, "memory_type": item.memory_type,
                    "content": item.content, "importance": item.importance, "created_at": item.created_at,
                    "session_id": item.session_id, "content_hash": _content_hash(item),
                    "embedding_signature": self.embedder.signature,
                }) for item, vector in zip(pending, vectors)],
            )
            indexed += len(pending)
        return indexed

    def search(self, query: str, user_id: str, limit: int = 10,
               memory_types: set[str] | None = None, session_id: str | None = None,
               min_importance: float = 0.0) -> dict[str, float]:
        must = [
            self.models.FieldCondition(key="user_id", match=self.models.MatchValue(value=user_id)),
            self.models.FieldCondition(key="memory_type", match=self.models.MatchAny(any=sorted(memory_types or INDEXED_MEMORY_TYPES))),
            self.models.FieldCondition(key="importance", range=self.models.Range(gte=min_importance)),
        ]
        if session_id:
            must.append(self.models.FieldCondition(key="session_id", match=self.models.MatchValue(value=session_id)))
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.embedder.encode_query(query),
            query_filter=self.models.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return {
            str(point.payload.get("memory_id", point.id)): float(point.score)
            for point in response.points
        }

    def delete(self, memory_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=self.models.PointIdsList(points=[memory_id]),
        )


class Neo4jGraphStore:
    """Neo4j 可选关系索引，实体/关系由调用方通过元数据显式提供。"""

    def __init__(self, uri: str | None = None, username: str | None = None,
                 password: str | None = None, database: str | None = None) -> None:
        from neo4j import GraphDatabase

        uri = uri or os.getenv("NEO4J_URI")
        username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD")
        if not uri or not password:
            raise ValueError("使用 Neo4j 需要设置 NEO4J_URI 和 NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

    def upsert(self, item: Any) -> None:
        entities = item.metadata.get("entities", [])
        relations = item.metadata.get("relations", [])
        with self.driver.session(database=self.database) as session:
            session.run(
                "MERGE (m:Memory {id: $id, user_id: $user_id}) "
                "SET m.content = $content, m.memory_type = $memory_type",
                id=item.id, user_id=item.user_id, content=item.content,
                memory_type=item.memory_type,
            )
            for entity in entities:
                name = str(entity).strip()
                if not name:
                    continue
                session.run(
                    "MERGE (e:Entity {name: $name, user_id: $user_id}) "
                    "WITH e MATCH (m:Memory {id: $memory_id, user_id: $user_id}) "
                    "MERGE (m)-[:MENTIONS]->(e)",
                    name=name, user_id=item.user_id, memory_id=item.id,
                )
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                source, target = str(relation.get("source", "")).strip(), str(relation.get("target", "")).strip()
                if not source or not target:
                    continue
                session.run(
                    "MERGE (a:Entity {name: $source, user_id: $user_id}) "
                    "MERGE (b:Entity {name: $target, user_id: $user_id}) "
                    "MERGE (a)-[r:RELATED_TO {kind: $kind, user_id: $user_id}]->(b) "
                    "WITH a, b MATCH (m:Memory {id: $memory_id, user_id: $user_id}) "
                    "MERGE (m)-[:SUPPORTS]->(a) MERGE (m)-[:SUPPORTS]->(b)",
                    source=source, target=target, kind=str(relation.get("type", "related_to")),
                    user_id=item.user_id, memory_id=item.id,
                )

    def search(self, query: str, user_id: str, limit: int = 10) -> dict[str, float]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (m:Memory {user_id: $user_id})-[:MENTIONS|SUPPORTS]->(e:Entity {user_id: $user_id}) "
                "WHERE toLower(e.name) CONTAINS toLower($query) "
                "RETURN DISTINCT m.id AS memory_id LIMIT $limit",
                user_id=user_id, query=query, limit=limit,
            )
            return {record["memory_id"]: 1.0 for record in result}

    def delete_memory_links(self, memory_id: str, user_id: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH (m:Memory {id: $id, user_id: $user_id}) DETACH DELETE m",
                id=memory_id, user_id=user_id,
            )

    def close(self) -> None:
        self.driver.close()

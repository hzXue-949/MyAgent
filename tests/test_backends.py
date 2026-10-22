import warnings
from unittest.mock import MagicMock, patch
from uuid import uuid4

from qdrant_client import QdrantClient, models

from my_agent.memory import MemoryItem
from my_agent.memory.backends import Neo4jGraphStore, QdrantVectorStore
from tests.helpers import IsolatedMemoryTestCase


class QdrantAdapterTests(IsolatedMemoryTestCase):
    def setUp(self):
        super().setUp()
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)

    def backend(self):
        with patch("qdrant_client.QdrantClient", return_value=self.client), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            return QdrantVectorStore(collection="unit_memory", embedder=self.embedder)

    def item(self, content="Python", user="alice", kind="semantic", importance=.5, session="s1"):
        return MemoryItem(str(uuid4()), user, kind, content, importance=importance, session_id=session)

    def test_model_collection_does_not_overwrite_legacy_collection(self):
        self.client.create_collection("unit_memory", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        backend = self.backend()
        self.assertNotEqual(backend.collection, "unit_memory")
        self.assertEqual(len(self.client.get_collections().collections), 2)

    def test_existing_collection_must_match_dimension(self):
        self.client.create_collection("unit_memory_" + self.embedder.signature, vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE))
        with self.assertRaises(ValueError):
            self.backend()

    def test_upsert_search_returns_qdrant_similarity_scores(self):
        backend = self.backend()
        expected = self.item()
        backend.upsert(expected)
        backend.upsert(self.item("食物"))
        result = backend.search("编程", "alice", 1)
        self.assertAlmostEqual(result[expected.id], 1.0)

    def test_payload_filters_user_type_session_importance(self):
        backend = self.backend()
        expected = self.item(importance=.9)
        for item in (expected, self.item(user="bob"), self.item(kind="episodic"), self.item(session="s2"), self.item(importance=.1)):
            backend.upsert(item)
        self.assertEqual(set(backend.search("编程", "alice", 10, {"semantic"}, "s1", .8)), {expected.id})

    def test_indexing_is_idempotent_and_content_changes_are_reencoded(self):
        backend = self.backend()
        item = self.item()
        self.assertEqual(backend.ensure_indexed([item]), 1)
        self.assertEqual(backend.ensure_indexed([item]), 0)
        self.assertEqual(len(self.embedder.document_batches), 1)
        item.content = "食物"
        self.assertEqual(backend.ensure_indexed([item]), 1)

    def test_delete_removes_point(self):
        backend = self.backend()
        item = self.item()
        backend.upsert(item)
        backend.delete(item.id)
        self.assertEqual(backend.search("编程", "alice"), {})


class Neo4jAdapterTests(IsolatedMemoryTestCase):
    def setUp(self):
        super().setUp()
        self.driver = MagicMock()
        with patch("neo4j.GraphDatabase.driver", return_value=self.driver):
            self.backend = Neo4jGraphStore(uri="neo4j://unit-test.invalid", password="unit-test-placeholder")
        self.session = self.driver.session.return_value.__enter__.return_value

    def test_graph_upsert_uses_explicit_entities_and_relations(self):
        item = MemoryItem("m1", "alice", "semantic", "Python 用于机器学习", metadata={
            "entities": ["Python", "机器学习"],
            "relations": [{"source": "Python", "target": "机器学习", "type": "USED_FOR"}],
        })
        self.backend.upsert(item)
        self.assertEqual(self.session.run.call_count, 4)
        for call in self.session.run.call_args_list:
            self.assertEqual(call.kwargs["user_id"], "alice")
        self.assertEqual(self.session.run.call_args.kwargs["kind"], "USED_FOR")

    def test_graph_search_and_delete_are_user_scoped(self):
        self.session.run.return_value = [{"memory_id": "m1"}]
        self.assertEqual(self.backend.search("Python", "alice"), {"m1": 1.0})
        self.assertEqual(self.session.run.call_args.kwargs["user_id"], "alice")
        self.backend.delete_memory_links("m1", "alice")
        self.assertEqual(self.session.run.call_args.kwargs, {"id": "m1", "user_id": "alice"})

    def test_graph_requires_credentials(self):
        with self.assertRaises(ValueError):
            Neo4jGraphStore()

    def test_close_closes_driver(self):
        self.backend.close()
        self.driver.close.assert_called_once()

"""使用已下载的真实 BGE/ONNX 模型，禁止网络和真实云数据库调用。"""

import math
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from my_agent.memory import MemoryConfig, MemoryManager
from my_agent.memory.embedding import LocalTransformerEmbedding


class RealBGEIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnxruntime
        onnxruntime.disable_telemetry_events()
        cls.patchers = [
            patch.dict(os.environ, {"EMBED_MODEL_TYPE": "local", "EMBED_MODEL_NAME": "BAAI/bge-small-zh-v1.5", "EMBED_LOCAL_FILES_ONLY": "true", "HF_HUB_OFFLINE": "1", "EMBED_DEVICE": "cpu"}, clear=True),
            patch("my_agent.memory.manager._load_dotenv"),
            patch("my_agent.memory.embedding.load_dotenv"),
            patch("socket.socket.connect", side_effect=AssertionError("集成测试禁止访问网络")),
        ]
        for patcher in cls.patchers:
            patcher.start()
            cls.addClassCleanup(patcher.stop)
        cls.embedder = LocalTransformerEmbedding()

    def test_real_model_outputs_finite_normalized_512_dimensional_vectors(self):
        vectors = self.embedder.encode_batch(["用户正在学习 Python 软件开发", "今天午餐吃面条"])
        self.assertEqual(len(vectors), 2)
        for vector in vectors:
            self.assertEqual(len(vector), 512)
            self.assertTrue(all(math.isfinite(value) for value in vector))
            self.assertAlmostEqual(sum(value * value for value in vector), 1., places=5)

    def test_real_semantic_similarity_prefers_related_document(self):
        docs = self.embedder.encode_batch(["用户正在学习 Python 软件开发，练习写代码。", "午餐吃西红柿鸡蛋面。"])
        query = self.embedder.encode_query("如何学习编程？")
        scores = [sum(a * b for a, b in zip(query, doc)) for doc in docs]
        self.assertGreater(scores[0], scores[1])

    def test_real_model_sqlite_memory_round_trip(self):
        with tempfile.TemporaryDirectory() as directory, patch("my_agent.memory.embedding.get_text_embedder", return_value=self.embedder):
            config = MemoryConfig(database_path=str(Path(directory) / "memory.sqlite3"))
            manager = MemoryManager("integration-user", config)
            expected = manager.add_memory("用户正在学习 Python 软件开发，练习写代码。", "semantic", .8)
            manager.add_memory("午餐吃西红柿鸡蛋面。", "semantic", .8)
            self.assertEqual(manager.retrieve_memories("如何学习编程？", limit=1)[0].id, expected)
            restored = MemoryManager("integration-user", config)
            self.assertEqual(restored.reindex_memories(), 0)
            self.assertEqual(restored.get_memory(expected).access_count, 1)

    def test_real_model_qdrant_memory_round_trip(self):
        from qdrant_client import QdrantClient
        from my_agent.memory.backends import QdrantVectorStore

        client = QdrantClient(":memory:")
        self.addCleanup(client.close)
        with tempfile.TemporaryDirectory() as directory, patch("my_agent.memory.embedding.get_text_embedder", return_value=self.embedder), patch.dict(os.environ, {"QDRANT_URL": "https://unit-test.invalid"}), patch("qdrant_client.QdrantClient", return_value=client), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            manager = MemoryManager("integration-user", MemoryConfig(database_path=str(Path(directory) / "memory.sqlite3")))
            self.assertIsInstance(manager.vector_store, QdrantVectorStore)
            expected = manager.add_memory("用户正在学习 Python 软件开发，练习写代码。", "semantic", .8)
            manager.add_memory("午餐吃西红柿鸡蛋面。", "semantic", .8)
            self.assertEqual(manager.retrieve_memories("如何学习编程？", limit=1)[0].id, expected)
            self.assertEqual(manager.reindex_memories(), 0)

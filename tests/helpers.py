import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from my_agent.memory import MemoryConfig, MemoryManager


class StubEmbedder:
    """仅用于单元测试的可预测向量，生产代码不使用该替身。"""

    dimensions = 3
    model_name = "unit-test-model"
    signature = "unit-test-model-v1"

    def __init__(self):
        self.document_batches = []
        self.queries = []

    @staticmethod
    def vector(text):
        if any(word in text.lower() for word in ("python", "编程", "代码")):
            return [1.0, 0.0, 0.0]
        if any(word in text for word in ("食物", "饮食", "吃饭")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def encode_batch(self, texts):
        self.document_batches.append(list(texts))
        return [self.vector(text) for text in texts]

    def encode_query(self, text):
        self.queries.append(text)
        return self.vector(text)


class IsolatedMemoryTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = str(Path(self.temp.name) / "memory.sqlite3")
        for patcher in (
            patch.dict(os.environ, {"EMBED_MODEL_TYPE": "local", "HF_HUB_OFFLINE": "1", "EMBED_LOCAL_FILES_ONLY": "true"}, clear=True),
            patch("my_agent.memory.manager._load_dotenv"),
            patch("my_agent.memory.embedding.load_dotenv"),
            patch("socket.socket.connect", side_effect=AssertionError("测试禁止访问网络")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.embedder = StubEmbedder()
        patcher = patch("my_agent.memory.embedding.get_text_embedder", return_value=self.embedder)
        patcher.start()
        self.addCleanup(patcher.stop)

    def manager(self, user_id="alice", **kwargs):
        config = kwargs.pop("config", MemoryConfig(database_path=self.database))
        manager = MemoryManager(user_id, config, **kwargs)
        if manager.store._memory_connection is not None:
            self.addCleanup(manager.store._memory_connection.close)
        return manager

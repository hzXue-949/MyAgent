import json
import time
from unittest.mock import Mock, patch

from my_agent.memory import MemoryConfig, MemoryItem, MemoryTool
from my_agent.memory.embedding import EmbeddingError
from my_agent.memory.manager import SQLiteMemoryStore, WorkingMemory, _lexical_similarity, _tokens

from tests.helpers import IsolatedMemoryTestCase


class SQLiteMemoryTests(IsolatedMemoryTestCase):
    def test_memory_database_retains_rows_across_connections(self):
        manager = self.manager(config=MemoryConfig(database_path=":memory:"))
        key = manager.add_memory("Python", "semantic")
        self.assertEqual(manager.get_memory(key).content, "Python")
        self.assertEqual(manager.retrieve_memories("编程", memory_types=["semantic"])[0].id, key)

    def test_file_database_survives_manager_recreation(self):
        first = self.manager()
        key = first.add_memory("Python", "semantic", metadata={"level": "beginner"})
        second = self.manager()
        self.assertEqual(second.get_memory(key).metadata, {"level": "beginner"})
        self.assertEqual(second.reindex_memories(), 0)

    def test_empty_type_filter_does_not_mean_all(self):
        manager = self.manager()
        manager.add_memory("Python", "semantic")
        self.assertEqual(manager.store.list("alice", set()), [])

    def test_store_delete_many_is_scoped_to_user(self):
        alice, bob = self.manager(), self.manager("bob")
        a = alice.add_memory("Python", "semantic")
        b = bob.add_memory("食物", "semantic")
        self.assertEqual(alice.store.delete_many("alice", [a, b]), 1)
        self.assertIsNotNone(bob.get_memory(b))
        self.assertEqual(alice.store.delete_many("alice", []), 0)

    def test_relative_database_path_uses_project_root(self):
        with patch("my_agent.memory.manager.PROJECT_ROOT", self.temp_path):
            store = SQLiteMemoryStore("nested/memory.sqlite3")
        self.assertEqual(store.database_path, str(self.temp_path / "nested/memory.sqlite3"))

    @property
    def temp_path(self):
        from pathlib import Path
        return Path(self.temp.name)

    def test_item_serialization_preserves_metadata_and_iso_times(self):
        item = MemoryItem("m1", "alice", "perceptual", "图片", created_at=0, updated_at=0, metadata={"modality": "image"})
        result = item.to_dict()
        self.assertEqual(result["created_at"], "1970-01-01T00:00:00+00:00")
        self.assertEqual(result["metadata"]["modality"], "image")


class WorkingMemoryTests(IsolatedMemoryTestCase):
    def test_working_memory_is_not_persistent(self):
        manager = self.manager()
        key = manager.add_memory("刚才讨论 Python", "working")
        self.assertIsNotNone(manager.get_memory(key))
        self.assertIsNone(self.manager().get_memory(key))
        self.assertEqual(self.embedder.document_batches, [])

    def test_ttl_expiration(self):
        working = WorkingMemory(capacity=2, ttl_minutes=1)
        with patch("my_agent.memory.manager.time.time", return_value=100):
            working.add(MemoryItem("a", "alice", "working", "Python", created_at=100))
        with patch("my_agent.memory.manager.time.time", return_value=161):
            self.assertEqual(working.list("alice"), [])

    def test_capacity_evicts_low_importance_then_oldest(self):
        working = WorkingMemory(capacity=2, ttl_minutes=100)
        now = time.time()
        for key, importance in (("a", .1), ("b", .9), ("c", .8)):
            working.add(MemoryItem(key, "alice", "working", key, importance=importance, created_at=now))
        self.assertEqual({item.id for item in working.list("alice")}, {"b", "c"})

    def test_working_only_search_excludes_persistent_items(self):
        manager = self.manager()
        manager.add_memory("Python", "semantic")
        key = manager.add_memory("Python", "working")
        self.assertEqual([m.id for m in manager.retrieve_memories("Python", memory_types=["working"])], [key])

    def test_session_filter_preserves_session_ids(self):
        manager = self.manager()
        a = manager.add_memory("Python", "working", session_id="s1")
        b = manager.add_memory("Python", "working", session_id="s2")
        self.assertEqual([m.id for m in manager.retrieve_memories("Python", session_id="s1")], [a])
        self.assertEqual(manager.get_memory(b).session_id, "s2")

    def test_lexical_tokens_and_empty_query(self):
        self.assertEqual(_tokens("学习 Python"), ["学习", "python"])
        self.assertGreater(_lexical_similarity("Python", "学习 Python"), _lexical_similarity("Python", "吃饭"))
        self.assertEqual(_lexical_similarity("", "Python"), 0)


class MemoryManagerTests(IsolatedMemoryTestCase):
    def test_user_isolation_applies_to_read_update_delete_clear(self):
        alice, bob = self.manager(), self.manager("bob")
        a = alice.add_memory("Python", "semantic")
        b = bob.add_memory("Python", "semantic")
        self.assertIsNone(alice.get_memory(b))
        self.assertFalse(alice.update_memory(b, "changed"))
        self.assertFalse(alice.remove_memory(b))
        self.assertEqual([m.id for m in alice.retrieve_memories("编程")], [a])
        self.assertEqual(alice.clear_all(), 1)
        self.assertIsNotNone(bob.get_memory(b))

    def test_invalid_add_parameters(self):
        manager = self.manager()
        for content, kind, importance in (("", "semantic", .5), ("x", "unknown", .5), ("x", "working", 1.1), ("x", "working", -.1)):
            with self.subTest(content=content, kind=kind, importance=importance), self.assertRaises(ValueError):
                manager.add_memory(content, kind, importance)

    def test_blank_user_is_rejected(self):
        with self.assertRaises(ValueError):
            self.manager(" ")

    def test_invalid_search_parameters(self):
        manager = self.manager()
        for kwargs in ({"query": ""}, {"query": "x", "limit": 0}, {"query": "x", "memory_types": ["unknown"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                manager.retrieve_memories(**kwargs)

    def test_semantic_search_uses_vectors_without_word_overlap(self):
        manager = self.manager()
        expected = manager.add_memory("Python", "semantic")
        manager.add_memory("食物", "semantic")
        self.assertEqual(manager.retrieve_memories("编程", limit=1)[0].id, expected)
        self.assertEqual(self.embedder.queries, ["编程"])

    def test_semantic_ranking_combines_vector_graph_and_importance(self):
        manager = self.manager()
        a = manager.add_memory("Python A", "semantic", importance=.5)
        b = manager.add_memory("Python B", "semantic", importance=.5)
        manager.graph_store = Mock()
        manager.graph_store.search.return_value = {a: 1.0}
        self.assertEqual(manager.retrieve_memories("编程", limit=1)[0].id, a)
        manager.graph_store.search.assert_called_once()
        self.assertNotEqual(a, b)

    def test_episodic_search_uses_vector_and_recency(self):
        manager = self.manager()
        old = manager.add_memory("Python old", "episodic", importance=.5)
        old_item = manager.get_memory(old)
        old_item.created_at -= 90 * 86400
        manager.store.save(old_item)
        recent = manager.add_memory("Python recent", "episodic", importance=.5)
        self.assertEqual(manager.retrieve_memories("编程", limit=1)[0].id, recent)

    def test_semantic_importance_changes_rank(self):
        manager = self.manager()
        manager.add_memory("Python low", "semantic", importance=.1)
        expected = manager.add_memory("Python high", "semantic", importance=.9)
        self.assertEqual(manager.retrieve_memories("编程", limit=1)[0].id, expected)

    def test_importance_and_session_filters(self):
        manager = self.manager()
        expected = manager.add_memory("Python", "semantic", .9, session_id="s1")
        manager.add_memory("Python", "semantic", .2, session_id="s1")
        manager.add_memory("Python", "semantic", .9, session_id="s2")
        self.assertEqual([m.id for m in manager.retrieve_memories("编程", min_importance=.8, session_id="s1")], [expected])

    def test_repeated_search_reuses_document_embeddings(self):
        manager = self.manager()
        manager.add_memory("Python", "semantic")
        for _ in range(2):
            manager.retrieve_memories("编程")
        self.assertEqual(len(self.embedder.document_batches), 1)

    def test_update_reencodes_changed_content(self):
        manager = self.manager()
        key = manager.add_memory("Python", "semantic")
        self.assertTrue(manager.update_memory(key, "食物", importance=.9, metadata={"edited": True}))
        self.assertEqual(self.embedder.document_batches[-1], ["食物"])
        self.assertEqual(manager.get_memory(key).metadata, {"edited": True})

    def test_access_count_is_persistent(self):
        manager = self.manager()
        key = manager.add_memory("Python", "semantic")
        manager.retrieve_memories("编程")
        self.assertEqual(self.manager().get_memory(key).access_count, 1)

    def test_reindex_skips_unchanged_and_force_rebuilds(self):
        manager = self.manager()
        manager.add_memory("Python", "semantic")
        self.assertEqual(manager.reindex_memories(), 0)
        self.assertEqual(manager.reindex_memories(force=True), 1)

    def test_search_backfills_missing_vectors_for_existing_raw_memory(self):
        manager = self.manager()
        item = MemoryItem("legacy-memory", "alice", "semantic", "Python")
        manager.store.save(item)
        self.assertEqual(manager.retrieve_memories("编程")[0].id, item.id)
        self.assertEqual(manager.reindex_memories(), 0)

    def test_empty_reindex_does_not_call_embedding_model(self):
        manager = self.manager()
        self.assertEqual(manager.reindex_memories(), 0)
        self.assertEqual(self.embedder.document_batches, [])

    def test_new_model_builds_separate_index(self):
        manager = self.manager()
        manager.add_memory("Python", "semantic")
        self.embedder.signature = "unit-test-model-v2"
        self.assertEqual(manager.reindex_memories(), 1)
        with manager.store._connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(DISTINCT signature) FROM memory_vectors").fetchone()[0], 2)

    def test_remove_cleans_local_vector_cache(self):
        manager = self.manager()
        key = manager.add_memory("Python", "semantic")
        self.assertTrue(manager.remove_memory(key))
        with manager.store._connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0], 0)

    def test_consolidate_moves_only_matching_important_working_memory(self):
        manager = self.manager()
        key = manager.add_memory("Python", "working", .9, session_id="s1")
        manager.add_memory("other", "working", .1, session_id="s1")
        manager.add_memory("other", "working", .9, session_id="s2")
        self.assertEqual(manager.consolidate("s1", .5), 1)
        item = manager.retrieve_memories("编程", memory_types=["episodic"])[0]
        self.assertEqual(item.metadata["consolidated_from"], key)
        self.assertEqual(item.session_id, "s1")
        self.assertIsNone(manager.get_memory(key))

    def test_forget_by_importance(self):
        manager = self.manager()
        manager.add_memory("Python", "semantic", .1)
        manager.add_memory("temporary", "working", .1)
        retained = manager.add_memory("食物", "semantic", .9)
        self.assertEqual(manager.forget_memories(threshold=.2), 2)
        self.assertIsNotNone(manager.get_memory(retained))

    def test_forget_by_time(self):
        manager = self.manager()
        key = manager.add_memory("Python", "episodic")
        item = manager.get_memory(key)
        item.updated_at -= 40 * 86400
        manager.store.save(item)
        self.assertEqual(manager.forget_memories(strategy="time_based", max_age_days=30), 1)

    def test_forget_by_capacity(self):
        manager = self.manager()
        manager.add_memory("Python low", "semantic", .1)
        key = manager.add_memory("Python high", "semantic", .9)
        self.assertEqual(manager.forget_memories(strategy="capacity_based", max_items=1), 1)
        self.assertIsNotNone(manager.get_memory(key))

    def test_invalid_forgetting_policy(self):
        manager = self.manager()
        for kwargs in ({"strategy": "unknown"}, {"strategy": "capacity_based"}, {"strategy": "capacity_based", "max_items": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                manager.forget_memories(**kwargs)

    def test_summary_and_stats(self):
        manager = self.manager()
        manager.add_memory("Python", "semantic", .9)
        manager.add_memory("temporary", "working", .1)
        result = manager.summary(limit=1)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["embedding_model"], "unit-test-model")
        self.assertEqual(len(result["recent_important"]), 1)

    def test_cloud_initialization_failure_uses_same_model_locally(self):
        with patch.dict("os.environ", {"QDRANT_URL": "https://unit-test.invalid"}), patch("my_agent.memory.backends.QdrantVectorStore", side_effect=OSError), self.assertWarns(RuntimeWarning):
            manager = self.manager()
        self.assertIs(manager.vector_store, manager.local_vector_store)

    def test_cloud_search_failure_uses_same_model_locally(self):
        manager = self.manager()
        key = manager.add_memory("Python", "semantic")
        manager.vector_store = Mock()
        manager.vector_store.search.side_effect = OSError("offline")
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(manager.retrieve_memories("编程")[0].id, key)

    def test_embedding_failure_is_not_hidden_by_fallback(self):
        manager = self.manager()
        manager.add_memory("Python", "semantic")
        manager.vector_store = Mock()
        manager.vector_store.search.side_effect = EmbeddingError("quota exhausted")
        with self.assertRaisesRegex(EmbeddingError, "quota exhausted"):
            manager.retrieve_memories("编程")

    def test_failed_add_keeps_raw_memory_and_reports_failure(self):
        manager = self.manager()
        self.embedder.encode_batch = Mock(side_effect=EmbeddingError("unavailable"))
        with self.assertRaisesRegex(RuntimeError, "原文已保存"):
            manager.add_memory("Python", "semantic")
        self.assertEqual(len(manager.store.list("alice")), 1)

    def test_graph_failure_does_not_discard_vector_results(self):
        manager = self.manager()
        key = manager.add_memory("Python", "semantic")
        manager.graph_store = Mock()
        manager.graph_store.search.side_effect = OSError
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(manager.retrieve_memories("编程")[0].id, key)


class PerceptualMemoryTests(IsolatedMemoryTestCase):
    def test_image_description_and_metadata_round_trip(self):
        manager = self.manager()
        metadata = {"modality": "image", "raw_data": "uploads/code.png", "mime_type": "image/png"}
        key = manager.add_memory("Python 代码截图", "perceptual", metadata=metadata)
        restored = self.manager().get_memory(key)
        self.assertEqual(restored.metadata, metadata)
        self.assertEqual(manager.retrieve_memories("截图", memory_types=["perceptual"])[0].id, key)
        self.assertEqual(self.embedder.document_batches, [])

    def test_audio_description_can_be_updated_and_deleted(self):
        manager = self.manager()
        key = manager.add_memory("会议录音", "perceptual", metadata={"modality": "audio", "raw_data": "meeting.wav"})
        self.assertTrue(manager.update_memory(key, "项目会议录音", metadata={"duration": 60}))
        self.assertEqual(manager.get_memory(key).metadata["duration"], 60)
        self.assertTrue(manager.remove_memory(key))

    def test_perceptual_can_be_disabled(self):
        manager = self.manager(enable_perceptual=False)
        with self.assertRaises(ValueError):
            manager.add_memory("图片", "perceptual")


class MemoryToolTests(IsolatedMemoryTestCase):
    def tool(self):
        tool = MemoryTool("alice", MemoryConfig(database_path=self.database))
        return tool

    def test_tool_add_search_update_remove(self):
        tool = self.tool()
        added = tool.run({"action": "add", "content": "Python", "memory_type": "semantic"})
        self.assertIn("记忆已添加", added)
        key = json.loads(tool.run({"action": "search", "query": "编程"}))[0]["id"]
        self.assertEqual(tool.run({"action": "update", "memory_id": key, "content": "Python 代码"}), "记忆已更新")
        self.assertEqual(tool.run({"action": "remove", "memory_id": key}), "记忆已删除")

    def test_tool_management_actions(self):
        tool = self.tool()
        tool.run({"action": "add", "content": "Python", "importance": .9})
        self.assertIn("整合 1", tool.run({"action": "consolidate"}))
        self.assertEqual(json.loads(tool.run({"action": "stats"}))["total"], 1)
        self.assertIn("recent_important", json.loads(tool.run({"action": "summary"})))
        self.assertIn("遗忘 1", tool.run({"action": "forget", "threshold": 1}))
        self.assertIn("清空 0", tool.run({"action": "clear_all"}))

    def test_invalid_tool_input_returns_readable_error(self):
        tool = self.tool()
        for params in ({"action": "unknown"}, {"action": "add"}, {"action": "update"}, {"action": "search", "query": ""}):
            with self.subTest(params=params):
                self.assertIn("记忆操作失败", tool.run(params))

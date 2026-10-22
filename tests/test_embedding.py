import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from my_agent.memory.embedding import (
    DashScopeEmbedding, EmbeddingError, LocalTransformerEmbedding,
    _normalize, get_text_embedder,
)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.dict(os.environ, {}, clear=True),
            patch("my_agent.memory.embedding.load_dotenv"),
            patch("socket.socket.connect", side_effect=AssertionError("测试禁止访问网络")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def cloud_embedder(self, **overrides):
        settings = {"EMBED_API_KEY": "unit-test-placeholder", "EMBED_BASE_URL": "https://unit-test.invalid/v1", "EMBED_DIMENSIONS": "2"}
        settings.update(overrides)
        with patch.dict(os.environ, settings), patch("openai.OpenAI") as factory:
            embedder = DashScopeEmbedding()
        return embedder, factory.return_value

    def test_factory_selects_local_without_cloud_credentials(self):
        self.assertIsInstance(get_text_embedder(), LocalTransformerEmbedding)

    def test_factory_selects_dashscope(self):
        with patch.dict(os.environ, {"EMBED_MODEL_TYPE": "dashscope", "EMBED_API_KEY": "unit-test-placeholder", "EMBED_BASE_URL": "https://unit-test.invalid/v1"}), patch("openai.OpenAI"):
            self.assertIsInstance(get_text_embedder(), DashScopeEmbedding)

    def test_unknown_provider_is_rejected(self):
        with patch.dict(os.environ, {"EMBED_MODEL_TYPE": "hashing"}), self.assertRaises(ValueError):
            get_text_embedder()

    def test_cloud_requires_credentials_and_endpoint(self):
        for settings in ({}, {"EMBED_API_KEY": "unit-test-placeholder"}, {"EMBED_BASE_URL": "https://unit-test.invalid/v1"}):
            with self.subTest(settings=list(settings)), patch.dict(os.environ, settings, clear=True), self.assertRaises(ValueError):
                DashScopeEmbedding()

    def test_cloud_accepts_dashscope_key_variable(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "unit-test-placeholder", "EMBED_BASE_URL": "https://unit-test.invalid/v1"}), patch("openai.OpenAI") as factory:
            DashScopeEmbedding()
        self.assertEqual(factory.call_args.kwargs["api_key"], "unit-test-placeholder")

    def test_cloud_batches_sorts_and_normalizes_results(self):
        embedder, client = self.cloud_embedder(EMBED_BATCH_SIZE="2")
        client.embeddings.create.side_effect = [
            SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[0., 4.]), SimpleNamespace(index=0, embedding=[3., 0.])]),
            SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[3., 4.])]),
        ]
        self.assertEqual(embedder.encode_batch(["a", "b", "c"]), [[1., 0.], [0., 1.], [.6, .8]])
        self.assertEqual(client.embeddings.create.call_count, 2)
        self.assertEqual(client.embeddings.create.call_args_list[0].kwargs["input"], ["a", "b"])
        self.assertEqual(client.embeddings.create.call_args.kwargs["encoding_format"], "float")

    def test_cloud_empty_batch_does_not_call_api(self):
        embedder, client = self.cloud_embedder()
        self.assertEqual(embedder.encode_batch([]), [])
        client.embeddings.create.assert_not_called()

    def test_cloud_rejects_missing_response_rows(self):
        embedder, client = self.cloud_embedder()
        client.embeddings.create.return_value = SimpleNamespace(data=[])
        with self.assertRaises(ValueError):
            embedder.encode("x")

    def test_cloud_failure_is_reported_without_switching_model(self):
        embedder, client = self.cloud_embedder()
        client.embeddings.create.side_effect = OSError("network failed")
        with self.assertRaises(EmbeddingError):
            embedder.encode_query("x")

    def test_signature_is_independent_of_credentials(self):
        first, _ = self.cloud_embedder(EMBED_API_KEY="first-test-placeholder")
        second, _ = self.cloud_embedder(EMBED_API_KEY="second-test-placeholder")
        self.assertEqual(first.signature, second.signature)

    def test_signature_changes_with_model_and_dimensions(self):
        first, _ = self.cloud_embedder()
        second, _ = self.cloud_embedder(EMBED_MODEL_NAME="another-model")
        third, _ = self.cloud_embedder(EMBED_DIMENSIONS="3")
        self.assertEqual(len({first.signature, second.signature, third.signature}), 3)

    def test_normalization_rejects_invalid_vectors(self):
        for vector, dimension in (([0., 0.], 2), ([1.], 2), ([math.nan, 1.], 2), ([math.inf, 1.], 2)):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                _normalize(vector, dimension)

    def test_local_query_instruction_is_not_added_to_documents(self):
        embedder = LocalTransformerEmbedding()
        model = Mock()
        model.embed.return_value = [SimpleNamespace(tolist=lambda: [1., 0.])]
        embedder.__dict__.update(model=model, dimensions=2)
        embedder.encode("文档")
        self.assertEqual(model.embed.call_args.args[0], ["文档"])
        embedder.encode_query("问题")
        self.assertEqual(model.embed.call_args.args[0], [embedder.query_instruction + "问题"])

    def test_local_empty_batch_does_not_load_model(self):
        embedder = LocalTransformerEmbedding()
        self.assertEqual(embedder.encode_batch([]), [])
        self.assertNotIn("model", embedder.__dict__)

    def test_local_rejects_configured_dimension_mismatch(self):
        with patch.dict(os.environ, {"EMBED_DIMENSIONS": "384"}):
            embedder = LocalTransformerEmbedding()
            with self.assertRaises(ValueError):
                _ = embedder.dimensions

    def test_local_rejects_unsupported_model(self):
        with patch.dict(os.environ, {"EMBED_MODEL_NAME": "unit-test/unsupported"}), self.assertRaises(ValueError):
            _ = LocalTransformerEmbedding().dimensions

    def test_local_model_error_is_explicit(self):
        embedder = LocalTransformerEmbedding()
        embedder.__dict__["model"] = Mock()
        embedder.model.embed.side_effect = OSError("offline")
        with self.assertRaises(EmbeddingError):
            embedder.encode("x")

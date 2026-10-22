"""统一文本嵌入：百炼 OpenAI 兼容 API 或本地 BGE/ONNX。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from functools import cached_property
from typing import Protocol

from dotenv import load_dotenv

from my_agent.paths import DATA_DIR, ENV_FILE


class EmbeddingError(RuntimeError):
    """嵌入服务故障；不应被误认为向量数据库故障或自动更换模型。"""


class TextEmbedder(Protocol):
    model_name: str
    dimensions: int
    signature: str

    def encode_batch(self, texts: list[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


def _signature(*settings: object) -> str:
    """模型、维度和查询指令不同的向量不能混用；密钥不参与身份标识。"""
    serialized = json.dumps(settings, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _normalize(vector: list[float], dimensions: int) -> list[float]:
    if len(vector) != dimensions or not all(math.isfinite(v) for v in vector):
        raise ValueError(f"Embedding 返回的向量无效，期望 {dimensions} 维")
    norm = math.sqrt(sum(v * v for v in vector))
    if not norm:
        raise ValueError("Embedding 返回了零向量")
    return [v / norm for v in vector]


class DashScopeEmbedding:
    """百炼文本向量 API；API Key 与对话模型配置独立。"""

    def __init__(self) -> None:
        from openai import OpenAI

        self.model_name = os.getenv("EMBED_MODEL_NAME") or "text-embedding-v4"
        self.dimensions = int(os.getenv("EMBED_DIMENSIONS") or "1024")
        api_key = os.getenv("EMBED_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("EMBED_BASE_URL")
        if not api_key or not base_url:
            raise ValueError("百炼嵌入需要 EMBED_API_KEY（或 DASHSCOPE_API_KEY）和 EMBED_BASE_URL")
        if self.dimensions <= 0:
            raise ValueError("EMBED_DIMENSIONS 必须大于 0")
        self.client = OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=float(os.getenv("EMBED_TIMEOUT") or "60"), max_retries=2,
        )
        # 同一网关的不同业务空间可共享模型，但换网关时仍隔离索引。
        self.signature = _signature("dashscope", base_url.rstrip("/"), self.model_name, self.dimensions)
        self.batch_size = min(10, max(1, int(os.getenv("EMBED_BATCH_SIZE") or "10")))

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset:offset + self.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name, input=batch,
                    dimensions=self.dimensions, encoding_format="float",
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"百炼 Embedding 调用失败（{type(exc).__name__}）；请检查 API Key、业务空间地址和模型额度"
                ) from exc
            rows = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in rows] != list(range(len(batch))):
                raise ValueError("Embedding API 返回的条数或索引与输入不一致")
            vectors.extend(_normalize(row.embedding, self.dimensions) for row in rows)
        return vectors

    def encode(self, text: str) -> list[float]:
        return self.encode_batch([text])[0]

    def encode_query(self, text: str) -> list[float]:
        return self.encode(text)


class LocalTransformerEmbedding:
    """本地预训练模型；默认中文 BGE small，不需要 API Key。"""

    def __init__(self) -> None:
        self.model_name = os.getenv("EMBED_MODEL_NAME") or "BAAI/bge-small-zh-v1.5"
        default_instruction = "为这个句子生成表示以用于检索相关文章：" if "bge-" in self.model_name and "-zh" in self.model_name else ""
        self.query_instruction = os.getenv("EMBED_QUERY_INSTRUCTION", default_instruction)
        self.batch_size = max(1, int(os.getenv("EMBED_BATCH_SIZE") or "32"))

    @cached_property
    def model(self):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("本地嵌入需要执行 uv sync --extra memory-local --extra memory-cloud") from exc
        if os.getenv("EMBED_DEVICE", "cpu") != "cpu":
            raise ValueError("当前本地 Embedding 使用 ONNX CPU 推理，请设置 EMBED_DEVICE=cpu")
        return TextEmbedding(
            model_name=self.model_name, cache_dir=str(DATA_DIR / "models"),
            providers=["CPUExecutionProvider"],
            local_files_only=os.getenv("EMBED_LOCAL_FILES_ONLY", "false").lower() == "true",
        )

    @cached_property
    def dimensions(self) -> int:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingError("本地嵌入缺少依赖，请执行 uv sync --extra memory-local --extra memory-cloud") from exc

        supported = {item["model"]: item["dim"] for item in TextEmbedding.list_supported_models()}
        if self.model_name not in supported:
            raise ValueError("此本地模型不在 FastEmbed 支持列表中，请选择受支持的文本模型")
        dimension = int(supported[self.model_name])
        configured = os.getenv("EMBED_DIMENSIONS")
        if configured and int(configured) != dimension:
            raise ValueError(f"本地模型输出 {dimension} 维，请移除或修正 EMBED_DIMENSIONS")
        return dimension

    @cached_property
    def signature(self) -> str:
        return _signature("local-onnx", self.model_name, self.dimensions, self.query_instruction)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = list(self.model.embed(texts, batch_size=self.batch_size))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"本地 Embedding 加载/编码失败（{type(exc).__name__}）；请检查模型缓存、下载网络和本地配置"
            ) from exc
        if len(vectors) != len(texts):
            raise ValueError("本地模型返回的向量条数与输入不一致")
        return [_normalize(vector.tolist(), self.dimensions) for vector in vectors]

    def encode(self, text: str) -> list[float]:
        return self.encode_batch([text])[0]

    def encode_query(self, text: str) -> list[float]:
        return self.encode(self.query_instruction + text)


def get_text_embedder() -> TextEmbedder:
    load_dotenv(ENV_FILE)
    provider = (os.getenv("EMBED_MODEL_TYPE") or "local").lower()
    if provider == "dashscope":
        return DashScopeEmbedding()
    if provider == "local":
        return LocalTransformerEmbedding()
    raise ValueError("EMBED_MODEL_TYPE 只支持 dashscope 或 local；不再使用哈希向量代替嵌入模型")

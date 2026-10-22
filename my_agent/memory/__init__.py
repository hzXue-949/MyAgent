"""第 8 章：记忆管理与可选存储适配。"""

from .manager import MemoryConfig, MemoryItem, MemoryManager, MemoryTool
from .embedding import DashScopeEmbedding, LocalTransformerEmbedding, get_text_embedder

__all__ = ["MemoryConfig", "MemoryItem", "MemoryManager", "MemoryTool",
           "DashScopeEmbedding", "LocalTransformerEmbedding", "get_text_embedder"]

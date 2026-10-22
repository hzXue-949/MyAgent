"""集中管理框架默认配置，并支持从环境变量加载（第 7.3.2 节）。"""

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Hello Agents 基础配置。"""

    # LLM 配置
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "auto"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 对话历史配置
    max_history_length: int = Field(default=100, ge=0)

    @classmethod
    def from_env(cls) -> "Config":
        """读取常用框架配置；未设置时使用字段默认值。"""
        return cls(
            default_model=os.getenv("LLM_MODEL_ID", "gpt-3.5-turbo"),
            default_provider=os.getenv("LLM_PROVIDER", "auto"),
            debug=os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "on"},
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.environ["MAX_TOKENS"]) if os.getenv("MAX_TOKENS") else None,
            max_history_length=int(os.getenv("MAX_HISTORY_LENGTH", "100")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """返回可序列化的配置字典。"""
        return self.model_dump()

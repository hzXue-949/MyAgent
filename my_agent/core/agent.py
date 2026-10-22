"""Agent 抽象基类（第 7.3.3 节）。"""

from abc import ABC, abstractmethod
from typing import Optional

from my_agent.core.config import Config
from my_agent.llm.client import HelloAgentsLLM
from my_agent.core.message import Message


class Agent(ABC):
    """为不同 Agent 范式提供统一的依赖、运行入口和历史管理。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config.from_env()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent 并返回最终文本；由具体 Agent 实现。"""
        raise NotImplementedError

    def add_message(self, message: Message) -> None:
        """将消息追加到历史，并按配置保留最近的消息。"""
        self._history.append(message)
        limit = self.config.max_history_length
        if limit == 0:
            self._history.clear()
        elif len(self._history) > limit:
            del self._history[:-limit]

    def clear_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """返回历史副本，避免外部直接改动内部列表。"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"

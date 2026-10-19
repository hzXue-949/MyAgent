"""统一的 OpenAI 兼容 LLM 客户端（第 4.1.3、7.2.2、7.2.3 节）。"""

import os
from typing import Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "auto",
        timeout: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **legacy_kwargs,
    ):
        """初始化客户端；支持智谱、Ollama 和通用 OpenAI 兼容端点。

        保留旧版 ``apiKey`` / ``baseUrl`` 参数名，避免第四章示例的调用方
        在升级后失效。
        """
        api_key = api_key or legacy_kwargs.pop("apiKey", None)
        base_url = base_url or legacy_kwargs.pop("baseUrl", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"不支持的参数: {unexpected}")

        self.provider = provider.lower() if provider else "auto"
        if self.provider == "auto":
            self.provider = os.getenv("LLM_PROVIDER", "auto").lower()
        if self.provider == "auto":
            self.provider = self._auto_detect_provider(api_key, base_url)

        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.model:
            raise ValueError("模型ID必须通过 model 参数或 LLM_MODEL_ID 配置。")
        if not self.api_key:
            raise ValueError(
                f"未找到 {self.provider} 的 API 密钥；请传入 api_key 或配置对应环境变量。"
            )
        if not self.base_url:
            raise ValueError("服务地址必须通过 base_url 或 LLM_BASE_URL 配置。")

        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    @staticmethod
    def _auto_detect_provider(
        api_key: Optional[str], base_url: Optional[str]
    ) -> str:
        """优先按服务地址识别，避免另一家已配置的密钥抢占选择。"""
        actual_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_url:
            parsed = urlparse(actual_url)
            host = (parsed.hostname or "").lower()
            port = parsed.port
            if "api-inference.modelscope.cn" in host:
                return "modelscope"
            if "open.bigmodel.cn" in host:
                return "zhipu"
            if host in {"localhost", "127.0.0.1", "::1"}:
                if port == 11434:
                    return "ollama"
                return "local"
            if host == "api.openai.com":
                return "openai"

        for env_name, provider in (
            ("MODELSCOPE_API_KEY", "modelscope"),
            ("OPENAI_API_KEY", "openai"),
            ("ZHIPU_API_KEY", "zhipu"),
        ):
            if os.getenv(env_name):
                return provider

        actual_key = api_key or os.getenv("LLM_API_KEY")
        if actual_key and actual_key.startswith("ms-"):
            return "modelscope"
        return "auto"

    def _resolve_credentials(
        self, api_key: Optional[str], base_url: Optional[str]
    ) -> tuple[str, str]:
        """按提供商解析密钥与地址，同时允许通用变量覆盖默认地址。"""
        key_envs = {
            "zhipu": "ZHIPU_API_KEY",
            "openai": "OPENAI_API_KEY",
            "modelscope": "MODELSCOPE_API_KEY",
        }
        default_urls = {
            "zhipu": "https://open.bigmodel.cn/api/paas/v4/",
            "openai": "https://api.openai.com/v1",
            "modelscope": "https://api-inference.modelscope.cn/v1/",
            "ollama": "http://localhost:11434/v1",
        }
        local_provider = self.provider in {"ollama", "vllm", "local"}
        resolved_key = (
            api_key
            or (os.getenv(key_envs[self.provider]) if self.provider in key_envs else None)
            or os.getenv("LLM_API_KEY")
        )
        if local_provider and not resolved_key:
            # OpenAI Python SDK 要求非空 key；这些兼容服务通常不验证它。
            resolved_key = self.provider
        resolved_url = base_url or os.getenv("LLM_BASE_URL") or default_urls.get(self.provider)
        return resolved_key, resolved_url

    def think(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                stream=True,
                **({"max_tokens": self.max_tokens} if self.max_tokens is not None else {}),
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


# --- 客户端使用示例 ---
if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes Python code.",
            },
            {"role": "user", "content": "写一个快速排序算法"},
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)

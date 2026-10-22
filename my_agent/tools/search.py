"""第 7.5.3 节：使用 Tavily 与 SerpApi 的可选多源搜索工具。"""

import json
import os
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from hello_agents.tools import Tool, ToolParameter, ToolRegistry


class MyAdvancedSearchTool(Tool):
    """聚合 Tavily 和 SerpApi 搜索结果，并在后端失败时继续使用可用来源。"""

    _BACKENDS = {"hybrid", "tavily", "serpapi"}

    def __init__(
        self,
        backend: str = "hybrid",
        tavily_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
        max_results: int = 5,
    ) -> None:
        if backend not in self._BACKENDS:
            raise ValueError(f"不支持的搜索后端：{backend}")
        if not 1 <= max_results <= 10:
            raise ValueError("max_results 必须在 1 到 10 之间")

        super().__init__(
            name="advanced_search",
            description=(
                "多源网页搜索工具。hybrid 模式会查询已配置的 Tavily 和 SerpApi，"
                "合并去重结果，并在一个来源失败时保留其他来源结果。"
            ),
        )
        self.backend = backend
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.max_results = max_results
        self._tavily_client: Any = None

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="需要查询的关键词或问题",
                required=True,
            )
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        query = parameters.get("input") or parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return self._to_json({"query": "", "results": [], "errors": ["搜索查询不能为空"]})

        query = query.strip()
        configured = {
            "tavily": bool(self.tavily_key),
            "serpapi": bool(self.serpapi_key),
        }
        backends = (
            [name for name in ("tavily", "serpapi") if configured[name]]
            if self.backend == "hybrid"
            else [self.backend]
        )
        if not backends or not configured.get(backends[0], False):
            missing = "TAVILY_API_KEY" if self.backend == "tavily" else "SERPAPI_API_KEY"
            if self.backend == "hybrid":
                message = "请配置 TAVILY_API_KEY 或 SERPAPI_API_KEY"
            else:
                message = f"请配置 {missing}"
            return self._to_json({"query": query, "results": [], "errors": [message]})

        results: list[dict[str, Any]] = []
        answers: list[dict[str, str]] = []
        errors: list[str] = []
        for backend in backends:
            try:
                response = (
                    self._search_tavily(query)
                    if backend == "tavily"
                    else self._search_serpapi(query)
                )
                if response.get("answer"):
                    answers.append({"source": backend, "text": response["answer"]})
                self._merge_results(results, response.get("results", []), backend)
            except Exception as exc:
                # 不将第三方 SDK 的异常内容带回提示词，避免泄漏请求细节或密钥。
                errors.append(f"{backend} 搜索失败（{type(exc).__name__}）")

        payload = {
            "query": query,
            "answers": answers,
            "results": results[: self.max_results],
            "errors": errors,
        }
        if not payload["results"] and not answers and not errors:
            payload["errors"].append("搜索服务没有返回结果")
        return self._to_json(payload)

    def _search_tavily(self, query: str) -> dict[str, Any]:
        if self._tavily_client is None:
            from tavily import TavilyClient

            self._tavily_client = TavilyClient(api_key=self.tavily_key)
        response = self._tavily_client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=self.max_results,
        )
        return {
            "answer": response.get("answer"),
            "results": [
                {
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                }
                for item in response.get("results", [])[: self.max_results]
            ],
        }

    def _search_serpapi(self, query: str) -> dict[str, Any]:
        from serpapi import SerpApiClient

        response = SerpApiClient(
            {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "gl": "cn",
                "hl": "zh-cn",
                "num": self.max_results,
            }
        ).get_dict()
        answer_box = response.get("answer_box", {})
        results = [
            {
                "title": item.get("title", ""),
                "content": item.get("snippet", ""),
                "url": item.get("link", ""),
            }
            for item in response.get("organic_results", [])[: self.max_results]
        ]
        return {"answer": answer_box.get("answer"), "results": results}

    @staticmethod
    def _merge_results(
        merged: list[dict[str, Any]], incoming: list[dict[str, Any]], source: str
    ) -> None:
        positions: dict[str, int] = {}
        for index, result in enumerate(merged):
            positions[MyAdvancedSearchTool._result_key(result)] = index

        for result in incoming:
            key = MyAdvancedSearchTool._result_key(result)
            if key in positions:
                existing = merged[positions[key]]
                if source not in existing["sources"]:
                    existing["sources"].append(source)
                if len(result.get("content", "")) > len(existing.get("content", "")):
                    existing["content"] = result["content"]
                if not existing.get("title"):
                    existing["title"] = result.get("title", "")
                continue

            positions[key] = len(merged)
            merged.append(
                {
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "sources": [source],
                }
            )

    @staticmethod
    def _result_key(result: dict[str, Any]) -> str:
        url = str(result.get("url") or "").strip()
        if url:
            try:
                parts = urlsplit(url)
                return urlunsplit(
                    (
                        parts.scheme.lower(),
                        parts.netloc.lower(),
                        parts.path.rstrip("/"),
                        parts.query,
                        "",
                    )
                )
            except ValueError:
                pass
        fallback = result.get("title") or result.get("content") or url
        return str(fallback).strip().casefold()

    @staticmethod
    def _to_json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)


def select_search_results(search_result: str) -> str:
    """保留搜索命中的标题、摘要和链接，作为后续工具的稳定输入。"""
    try:
        payload = json.loads(search_result)
    except (TypeError, json.JSONDecodeError):
        return MyAdvancedSearchTool._to_json(
            {"query": "", "results": [], "errors": ["搜索结果不是有效 JSON"]}
        )
    if not isinstance(payload, dict):
        return MyAdvancedSearchTool._to_json(
            {"query": "", "results": [], "errors": ["搜索结果格式不正确"]}
        )
    results = payload.get("results")
    answers = payload.get("answers")
    errors = payload.get("errors")
    return MyAdvancedSearchTool._to_json(
        {
            "query": payload.get("query", ""),
            "answers": answers if isinstance(answers, list) else [],
            "results": results[:5] if isinstance(results, list) else [],
            "errors": errors if isinstance(errors, list) else [],
        }
    )


def format_search_digest(search_result: str) -> str:
    """将统一格式的搜索结果转换为便于阅读的摘要文本。"""
    try:
        payload = json.loads(search_result)
    except (TypeError, json.JSONDecodeError):
        return "搜索结果无法解析。"
    if not isinstance(payload, dict):
        return "搜索结果格式不正确。"

    sections = [f"搜索主题：{payload.get('query', '')}"]
    answers = payload.get("answers")
    results = payload.get("results")
    errors = payload.get("errors")
    answers = answers if isinstance(answers, list) else []
    results = results if isinstance(results, list) else []
    errors = errors if isinstance(errors, list) else []

    for answer in answers:
        if not isinstance(answer, dict):
            continue
        sections.append(f"直接回答（{answer.get('source', '未知来源')}）：{answer.get('text', '')}")
    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue
        sources = ", ".join(result.get("sources", []))
        sections.append(
            f"{index}. {result.get('title', '无标题')}\n"
            f"   {result.get('content', '')}\n"
            f"   来源：{result.get('url', '')}（{sources}）"
        )
    if errors:
        sections.append("搜索提示：" + "；".join(str(error) for error in errors))
    if not results and not answers and not errors:
        sections.append("没有找到相关结果。")
    return "\n\n".join(sections)


def register_advanced_search(
    registry: ToolRegistry,
    backend: str = "hybrid",
    tavily_key: Optional[str] = None,
    serpapi_key: Optional[str] = None,
    max_results: int = 5,
) -> ToolRegistry:
    """向现有注册表添加多源搜索和两个摘要处理函数。"""
    registry.register_tool(
        MyAdvancedSearchTool(
            backend=backend,
            tavily_key=tavily_key,
            serpapi_key=serpapi_key,
            max_results=max_results,
        )
    )
    registry.register_function(
        name="select_search_results",
        description="从多源搜索结果中提取前五条标题、摘要和链接。",
        func=select_search_results,
    )
    registry.register_function(
        name="format_search_digest",
        description="将搜索结果整理为包含标题、摘要、链接和来源的文本摘要。",
        func=format_search_digest,
    )
    return registry


def create_advanced_search_registry(
    backend: str = "hybrid",
    tavily_key: Optional[str] = None,
    serpapi_key: Optional[str] = None,
    max_results: int = 5,
) -> ToolRegistry:
    """创建一个包含多源搜索与结果整理工具的注册表。"""
    return register_advanced_search(
        ToolRegistry(), backend, tavily_key, serpapi_key, max_results
    )

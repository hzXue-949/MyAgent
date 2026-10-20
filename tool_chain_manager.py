"""第 7.5.4 节：使用上下文模板串联多个工具。"""

from typing import Any, Optional

from hello_agents import ToolRegistry


class ToolChain:
    """按顺序运行工具，并把每一步输出写入后续模板的上下文。"""

    def __init__(self, name: str, description: str) -> None:
        if not name.strip():
            raise ValueError("工具链名称不能为空")
        self.name = name
        self.description = description
        self.steps: list[dict[str, str]] = []

    def add_step(
        self, tool_name: str, input_template: str, output_key: Optional[str] = None
    ) -> None:
        if not tool_name.strip():
            raise ValueError("工具名称不能为空")
        key = output_key or f"step_{len(self.steps)}_result"
        if not key.strip():
            raise ValueError("输出键名不能为空")
        self.steps.append(
            {
                "tool_name": tool_name,
                "input_template": input_template,
                "output_key": key,
            }
        )

    def execute(
        self,
        registry: ToolRegistry,
        initial_input: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        if not self.steps:
            return f"工具链“{self.name}”没有执行步骤。"

        values = dict(context or {})
        values["input"] = initial_input
        known_tools = set(registry.list_tools())

        for index, step in enumerate(self.steps, start=1):
            tool_name = step["tool_name"]
            if tool_name not in known_tools:
                return f"工具链“{self.name}”第 {index} 步找不到工具“{tool_name}”。"
            try:
                tool_input = step["input_template"].format_map(values)
            except KeyError as exc:
                return f"工具链“{self.name}”第 {index} 步缺少模板变量：{exc.args[0]}。"

            result = registry.execute_tool(tool_name, tool_input)
            if result.startswith(("错误：执行工具", "错误：未找到名为")):
                return f"工具链“{self.name}”第 {index} 步失败：{result}"
            values[step["output_key"]] = result

        return values[self.steps[-1]["output_key"]]


class ToolChainManager:
    """管理可复用工具链并通过名称执行。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.chains: dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain, *, replace: bool = False) -> None:
        if chain.name in self.chains and not replace:
            raise ValueError(f"工具链“{chain.name}”已经注册")
        self.chains[chain.name] = chain

    def execute_chain(
        self,
        chain_name: str,
        input_data: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        chain = self.chains.get(chain_name)
        if chain is None:
            return f"工具链“{chain_name}”不存在。"
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> list[str]:
        return list(self.chains)


def create_search_digest_chain() -> ToolChain:
    """创建三步搜索流程：搜索 -> 筛选结果 -> 格式化摘要。"""
    chain = ToolChain(
        name="search_digest",
        description="搜索网页、筛选结果并生成带来源的摘要。",
    )
    chain.add_step("advanced_search", "{input}", "search_results")
    chain.add_step("select_search_results", "{search_results}", "selected_results")
    chain.add_step("format_search_digest", "{selected_results}", "digest")
    return chain

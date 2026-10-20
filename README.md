# MyAgent：Hello-Agents 学习与本地适配

这是跟随 [Datawhale《Hello-Agents》](https://hello-agents.datawhale.cc/) 学习 Agent 基础范式的练习仓库。仓库里的不少结构和思路来自教程示例；这里不把教程内容说成原创框架。下面列出实际覆盖的章节、本地改动和暂未实现的内容。

## 覆盖范围

| 教程内容 | 对应代码 |
| --- | --- |
| 第 4 章：ReAct、Plan-and-Solve、Reflection | `ReAct.py`、`Plan_and_solve.py`、`Reflection.py`、`tools.py` |
| 第 7.2 节：ModelScope 扩展示例、Ollama、本地提供商识别 | `my_llm.py`、`local_model.py`、`llm_client.py` |
| 第 7.3 节：消息、配置和 Agent 基类 | `message.py`、`config.py`、`agent.py` |
| 第 7.4.1–7.4.5 节：简单对话、ReAct、Reflection、Plan-and-Solve、函数调用 | `my_*_agent.py` |
| 第 7.5.1 节：工具抽象与注册表 | `hello-agents==0.1.1` 提供的 `Tool`、`ToolParameter`、`ToolRegistry` |
| 第 7.5.2 节：自定义计算器 | `my_calculator_tool.py` |
| 第 7.5.3 节：多源搜索与结果整理 | `my_advanced_search.py` |
| 第 7.5.4 节：工具链与并行执行 | `tool_chain_manager.py`、`async_tool_executor.py` |

## 相对教程示例的本地改动

- **模型连接：** `llm_client.py` 保留原有 `think(messages)` 调用方式和旧参数名，并增加明确指定或自动识别服务提供商的逻辑。自动识别主要根据服务地址判断，避免同时配置智谱密钥和本地 Ollama 地址时选错服务。当前主要使用场景是智谱和 Ollama；自动识别的是“连接哪个服务”，不会替你挑选最合适的模型。模型仍由 `LLM_MODEL_ID` 或代码参数指定。
- **教程扩展示例：** `my_llm.py` 保留了一个单独的 ModelScope Provider 练习示例；它不是本地默认配置路径。
- **本地模型：** `local_model.py` 是独立的 Ollama 调用示例，显式连接本机 `127.0.0.1:11434`，不依赖云端密钥。
- **框架基础类：** `message.py`、`config.py` 和 `agent.py` 提供消息校验与转换、环境变量配置、Agent 抽象接口和有长度限制的对话历史。这些是按教程内容在本仓库落地的练习代码。
- **7.4.3 和 7.4.4：** 在 Hello-Agents 已有 Agent 模式上实现可覆盖的提示词、计划解析和逐步执行逻辑。
- **7.4.5：** 项目锁定的 `hello-agents==0.1.1` 没有教程所需的 `FunctionCallAgent`，因此 `my_function_call_agent.py` 在本地把 `ToolRegistry` 工具转换成 OpenAI tools schema，执行模型返回的工具调用，并把结果交回模型。服务端和所选模型也必须支持原生 `tools` 调用。
- **7.5.1：** `Tool`、`ToolParameter` 和 `ToolRegistry` 直接复用锁定版本提供的实现，避免与 7.4 节已经使用的注册表产生两套类型。
- **7.5.2：** `my_calculator_tool.py` 用受限 AST 解释器计算表达式，不调用 `eval`，并提供注册函数供 Agent 和工具链复用。
- **7.5.3：** `my_advanced_search.py` 可组合 Tavily 与 SerpApi，统一标题、摘要、链接和来源字段，按 URL 去重；一个后端失败时会保留另一个后端的结果。SerpApi 使用基础依赖，Tavily 是可选依赖。
- **7.5.4：** `tool_chain_manager.py` 提供模板变量传递和三步搜索摘要示例；`async_tool_executor.py` 用有界线程池并行运行独立工具，并按输入顺序返回结果。
- **第 4 章的健壮性处理：** 无效的 ReAct Action 会作为错误观察交回模型；Plan-and-Solve 在空计划时返回空结果，避免变量未初始化错误。

## 尚未实现

- 第 8 章及之后的内容尚未覆盖。
- 没有根据任务自动选择或比较模型的功能；提供商自动识别不等于模型自动选择。
- 这是章节练习代码，不是完整的生产级 Agent 框架。

## 运行

需要 Python 3.10 及以上版本。安装依赖：

```bash
cp .env.example .env
uv sync
```

按需配置 `.env`。智谱示例：

```dotenv
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=你的密钥
LLM_MODEL_ID=你的模型ID
```

本地 Ollama 示例：先启动 Ollama 并下载模型，再运行独立示例：

```bash
ollama pull llama3:8b
uv run python local_model.py "请用中文简单介绍一下自己"
```

第四章的搜索版 ReAct 示例使用 `.env` 中的 `SERPAPI_API_KEY`。第 7.5 节的多源搜索可配置 SerpApi、Tavily 或两者；使用 Tavily 时需在虚拟环境中安装可选包 `tavily-python`。其他示例不需要搜索密钥。

```bash
uv pip install --python .venv/bin/python tavily-python
```

## 第 7.5 节工具系统示例

```python
import asyncio

from async_tool_executor import AsyncToolExecutor
from my_advanced_search import create_advanced_search_registry
from my_calculator_tool import register_calculator
from tool_chain_manager import ToolChainManager, create_search_digest_chain

registry = create_advanced_search_registry()
register_calculator(registry)

# 三步工具链：网页搜索 -> 筛选结果 -> 格式化摘要
chains = ToolChainManager(registry)
chains.register_chain(create_search_digest_chain())
print(chains.execute_chain("search_digest", "Python 3.14 新特性"))

# 适合并行执行的独立任务
async def run_parallel():
    with AsyncToolExecutor(registry, max_workers=4) as executor:
        return await executor.execute_tools_parallel([
            {"tool_name": "my_calculator", "input_data": "sqrt(16) + 2 * 3"},
            {"tool_name": "advanced_search", "input_data": "Python asyncio"},
        ])

print(asyncio.run(run_parallel()))
```

## 主要入口

- `ReAct.py`、`Plan_and_solve.py`、`Reflection.py`：第 4 章范式示例。
- `local_model.py`：Ollama 本地模型示例。
- `my_simple_agent.py`、`my_react_agent.py`、`my_reflection_agent.py`、`my_plan_solve_agent.py`、`my_function_call_agent.py`：第 7.4 节 Agent 练习。
- `my_calculator_tool.py`、`my_advanced_search.py`、`tool_chain_manager.py`、`async_tool_executor.py`：第 7.5 节工具系统练习。
- `test_simple_agent.py`：手动运行的对话与工具调用示例。

教程原文与章节说明见 [Datawhale《Hello-Agents》](https://hello-agents.datawhale.cc/)。

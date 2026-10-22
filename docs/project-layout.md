# 目录与迁移说明

`my_agent/` 按功能组织代码，`examples/` 按教程章节组织手动运行入口。Python 模块名统一使用小写和下划线，类名保持原样。

## 各目录职责

| 目录 | 作用 |
| --- | --- |
| `my_agent/core/` | 第 7.3 节本地 Config、Message、Agent 基类 |
| `my_agent/llm/` | 本地 `think()` 客户端、Provider 识别和 ModelScope 扩展 |
| `my_agent/agents/` | 基于 Hello-Agents SDK 的简单对话、ReAct、Reflection、Plan-and-Solve 和函数调用适配 |
| `my_agent/tools/` | 自定义计算、搜索、工具链、并行执行及第 4 章旧工具执行器 |
| `my_agent/memory/` | 记忆管理器、工具、百炼/本地 Embedding，以及 SQLite/Qdrant/Neo4j 适配 |
| `examples/chapter04/` | 第 4 章自包含范式实现和运行入口，保留其学习过程 |
| `examples/chapter07/` | 第 7 章模型、Agent 与工具的手动演示 |
| `examples/chapter08/` | 第 8 章记忆系统的手动演示 |
| `data/` | 运行时生成的本地数据，已加入 Git 忽略 |

第 7.3 节本地基础类和第 7.4 节 SDK 适配沿用既有依赖关系：`my_agent.core.Agent` 使用本地客户端；`my_agent.agents` 中的适配类继承 SDK Agent，并继续使用 SDK 的 Config、Message 和 HelloAgentsLLM。

## 旧文件到新位置

| 原根目录文件 | 新位置 |
| --- | --- |
| `agent.py` | [`my_agent/core/agent.py`](../my_agent/core/agent.py) |
| `config.py` | [`my_agent/core/config.py`](../my_agent/core/config.py) |
| `message.py` | [`my_agent/core/message.py`](../my_agent/core/message.py) |
| `llm_client.py` | [`my_agent/llm/client.py`](../my_agent/llm/client.py) |
| `my_llm.py` | [`my_agent/llm/modelscope.py`](../my_agent/llm/modelscope.py) |
| `my_simple_agent.py` | [`my_agent/agents/simple.py`](../my_agent/agents/simple.py) |
| `my_react_agent.py` | [`my_agent/agents/react.py`](../my_agent/agents/react.py) |
| `my_reflection_agent.py` | [`my_agent/agents/reflection.py`](../my_agent/agents/reflection.py) |
| `my_plan_solve_agent.py` | [`my_agent/agents/plan_solve.py`](../my_agent/agents/plan_solve.py) |
| `my_function_call_agent.py` | [`my_agent/agents/function_call.py`](../my_agent/agents/function_call.py) |
| `my_calculator_tool.py` | [`my_agent/tools/calculator.py`](../my_agent/tools/calculator.py) |
| `my_advanced_search.py` | [`my_agent/tools/search.py`](../my_agent/tools/search.py) |
| `tool_chain_manager.py` | [`my_agent/tools/chain.py`](../my_agent/tools/chain.py) |
| `async_tool_executor.py` | [`my_agent/tools/async_executor.py`](../my_agent/tools/async_executor.py) |
| `tools.py` | [`my_agent/tools/legacy.py`](../my_agent/tools/legacy.py) |
| `memory_system.py` | [`my_agent/memory/manager.py`](../my_agent/memory/manager.py) |
| `memory_backends.py` | [`my_agent/memory/backends.py`](../my_agent/memory/backends.py) |
| `ReAct.py` | [`examples/chapter04/react.py`](../examples/chapter04/react.py) |
| `Plan_and_solve.py` | [`examples/chapter04/plan_and_solve.py`](../examples/chapter04/plan_and_solve.py) |
| `Reflection.py` | [`examples/chapter04/reflection.py`](../examples/chapter04/reflection.py) |
| `planner_ab_test.py` | [`examples/chapter04/planner_comparison.py`](../examples/chapter04/planner_comparison.py) |
| `local_model.py` | [`examples/chapter07/local_model.py`](../examples/chapter07/local_model.py) |
| `test.py` | [`examples/chapter07/basic_agent.py`](../examples/chapter07/basic_agent.py) |
| `test_simple_agent.py` | [`examples/chapter07/simple_agent.py`](../examples/chapter07/simple_agent.py) |

## 新的运行与导入方式

在项目根目录运行：

```bash
uv run python -m examples.chapter07.simple_agent
uv run python -m examples.chapter07.calculator
uv run python -m examples.chapter07.parallel_tools
uv run --extra memory-local --extra memory-cloud python -m examples.chapter08.memory_demo
```

旧根目录脚本已迁移，导入路径也相应调整：

```python
from my_agent.agents.simple import MySimpleAgent
from my_agent.llm.client import HelloAgentsLLM
from my_agent.tools.calculator import create_calculator_registry
from my_agent.memory import MemoryConfig, MemoryManager, MemoryTool
```

可复用模块里的演示已提取为 `examples/chapter04/search.py`、`examples/chapter07/llm_client.py`、`examples/chapter07/calculator.py` 和 `examples/chapter08/memory_demo.py`。另提供 `parallel_tools.py` 和 `search_digest.py`，直接运行 README 原有的工具链示例。

示例导入时只加载定义，模型调用集中在运行入口。`.env` 留在项目根目录，`my_agent/paths.py` 统一提供配置和本地数据路径。

# Hello-Agents 第四章：智能体经典范式构建

照 [《Hello-Agents》第四章](https://hello-agents.datawhale.cc/#/./chapter4/第四章%20智能体经典范式构建) 的步骤，从零实现三种经典智能体范式。

| 范式 | 文件 | 核心思想 | 适用场景 |
| --- | --- | --- | --- |
| **ReAct** | `ReAct.py` | 思考-行动-观察循环，边想边做 | 需外部工具、时效性信息 |
| **Plan-and-Solve** | `Plan_and_solve.py` | 先规划后执行，两阶段解耦 | 多步推理、结构化任务 |
| **Reflection** | `Reflection.py` | 执行-反思-优化迭代 | 对质量要求高、时效宽松 |

## 环境准备

已使用 `uv` + Python 3.11 初始化完成：

```bash
uv sync --python 3.11
```

依赖：`openai`、`python-dotenv`、`google-search-results`（SerpApi）。

## 配置密钥

```bash
cp .env.example .env   # 已自动创建，直接编辑即可
```

云端模型可以填写以下通用配置。使用智谱时，也可以用 `ZHIPU_API_KEY`，客户端会识别智谱服务地址并使用智谱专属密钥：

```bash
LLM_API_KEY="你的密钥"
LLM_MODEL_ID="模型ID"
LLM_BASE_URL="服务地址"
```

运行 `ReAct.py` 还需 SerpApi 密钥，到 [serpapi.com](https://serpapi.com/) 注册免费账号获取：

```bash
SERPAPI_API_KEY="你的SerpApi密钥"
```

> `Plan_and_solve.py` 和 `Reflection.py` 不需要 SerpApi。

## 运行

```bash
# 4.1 基础 LLM 客户端自测（不需要 SerpApi）
uv run python llm_client.py

# 4.2 ReAct：搜索“华为最新手机”
uv run python ReAct.py

# 4.3 Plan-and-Solve：水果店应用题
uv run python Plan_and_solve.py

# 4.4 Reflection：素数函数迭代优化
uv run python Reflection.py
```

## 第 7.2.2 节：本地模型调用

按[教程第 7.2.2 节](https://hello-agents.datawhale.cc/#/./chapter7/第七章%20构建你的Agent框架?id=_722-本地模型调用)的 Ollama 路线运行 `llama3:8b`，模型下载约 4.7 GB。Apple Silicon 可使用 GPU 加速，不需要安装 CUDA 或 vLLM。

首次安装、启动服务和下载模型：

```bash
brew install ollama
brew services run ollama
ollama pull llama3:8b
```

服务默认仅监听 `127.0.0.1:11434`。这里的 `brew services run` 不设置登录自启，重启电脑后需要重新运行；也可以在单独终端中执行 `ollama serve`。

直接在终端聊天：

```bash
ollama run llama3:8b
# 输入 /bye 退出聊天
```

通过项目中的 HelloAgentsLLM 流式调用：

```bash
uv run python local_model.py
uv run python local_model.py "请用中文解释 Python 中 super().__init__ 的作用。"
```

示例显式使用以下配置，不读取或修改现有 `.env`，不会切换其他练习代码的云端模型：

```python
from hello_agents import HelloAgentsLLM

llm = HelloAgentsLLM(
    provider="ollama",
    model="llama3:8b",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",  # SDK 要求非空；本地服务不需要真实密钥
)
```

查看模型和释放资源：

```bash
ollama list                 # 已下载模型
ollama ps                   # 已加载模型，以及 CPU/GPU 使用情况
ollama stop llama3:8b        # 卸载内存中的模型，保留下载文件
brew services stop ollama   # 停止后台服务，不删除模型
```

模型默认保存在 `~/.ollama/models`。下载完成后，本地推理无需云端 API 密钥。

## 第 7.2.3 节：自动检测模型提供商

`llm_client.HelloAgentsLLM` 保持原有的 `think(messages)` 用法，初始化时可以指定 `provider`，也可以让它先根据 `LLM_BASE_URL` 的服务域名/本地端口推断，再检查专属密钥和通用密钥格式。这样即使同时配置了智谱密钥和 Ollama 地址，也会按当前服务地址选 Ollama。支持 `zhipu`、`ollama`、`openai`、`modelscope` 及其他 OpenAI 兼容地址。

Ollama 自动识别示例（无需云端 key）：

```dotenv
LLM_MODEL_ID=llama3:8b
LLM_BASE_URL=http://127.0.0.1:11434/v1
```

智谱自动识别示例：

```dotenv
ZHIPU_API_KEY=你的智谱密钥
LLM_MODEL_ID=glm-4-flash
```

也可通过 `LLM_PROVIDER` 手动指定提供商，显式传入构造参数 `provider` 的优先级最高。自动识别时服务地址优先于密钥；如果想覆盖自动判断，请显式设置 `LLM_PROVIDER` 或传入 `provider`。

## 第 7.3 节：框架接口

新增三个可直接复用的框架基础组件：

```python
from agent import Agent
from config import Config
from message import Message
```

- `Message`：Pydantic 校验的标准消息对象，可用 `to_dict()` 转为模型 API 消息格式。
- `Config`：集中保存模型、温度、日志和历史长度等设置；`Config.from_env()` 可从环境变量读取。
- `Agent`：抽象基类定义统一的 `run()` 入口、LLM/配置依赖和对话历史管理，具体范式通过子类实现 `run()`。

## 第 7.4.3–7.4.5 节：框架化 Agent

三个 Agent 使用同一个 `hello_agents.HelloAgentsLLM`，都通过 `run(text)` 执行任务：

```python
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool

from my_reflection_agent import MyReflectionAgent
from my_plan_solve_agent import MyPlanAndSolveAgent
from my_function_call_agent import MyFunctionCallAgent

llm = HelloAgentsLLM()

# 7.4.3：生成初稿，再反思和优化；custom_prompts 可覆盖任一默认提示词
reflection = MyReflectionAgent(name="反思助手", llm=llm, max_iterations=2)
print(reflection.run("写一段简洁的项目介绍"))

# 7.4.4：先生成步骤列表，再按顺序执行
planner = MyPlanAndSolveAgent(name="规划助手", llm=llm)
print(planner.run("三天分别卖出15、30、25个苹果，总共卖出多少？"))

# 7.4.5：模型按 OpenAI tools 协议选择并调用工具
registry = ToolRegistry()
registry.register_tool(CalculatorTool())
function_agent = MyFunctionCallAgent(name="函数调用助手", llm=llm, tool_registry=registry)
print(function_agent.run("计算 15 * 8 + 32"))
```

`MyReflectionAgent` 和 `MyPlanAndSolveAgent` 使用框架提供的 Agent 接口和历史管理，本地代码实现反思循环、计划解析与逐步执行，并支持逐项覆盖提示词。项目锁定的 `hello-agents==0.1.1` 尚未包含 `FunctionCallAgent`，所以 `MyFunctionCallAgent` 在本地实现 schema 生成、工具执行和多轮工具结果回传。7.4.5 需要所选模型服务支持 OpenAI 兼容的原生 `tools` 调用。

## 文件说明

```
.
├── llm_client.py       # 4.1.3 / 7.2.2 / 7.2.3  统一客户端、流式调用和提供商识别
├── tools.py            # 4.2.2  search() 搜索工具 + ToolExecutor 工具执行器
├── ReAct.py            # 4.2.3  ReActAgent：Thought/Action/Observation 循环
├── Plan_and_solve.py   # 4.3    Planner（规划器）+ Executor（执行器）
├── Reflection.py       # 4.4    Memory（短期记忆）+ ReflectionAgent
├── local_model.py      # 7.2.2  HelloAgentsLLM + 本机 Ollama 流式调用
├── my_llm.py           # 7.2.1  ModelScope Provider 扩展示例
├── message.py          # 7.3.1  标准消息与 API 字典转换
├── config.py           # 7.3.2  Pydantic 配置与环境变量加载
├── agent.py            # 7.3.3  Agent 抽象基类与历史管理
├── my_simple_agent.py  # 7.4.1  简单对话与提示词工具调用
├── my_react_agent.py   # 7.4.2  框架化 ReAct Agent
├── my_reflection_agent.py # 7.4.3  反思与迭代优化
├── my_plan_solve_agent.py # 7.4.4  规划后逐步执行
├── my_function_call_agent.py # 7.4.5  原生函数调用 Agent
├── test_simple_agent.py # 7.4.1  手动运行示例
├── test.py             # HelloAgents 基础使用示例
├── planner_ab_test.py  # Planner 提示词对照实验
├── pyproject.toml      # 项目依赖声明
├── .env                # 你的密钥（已被 .gitignore 忽略）
└── .env.example        # 密钥模板
```

## 与教程原文的差异

第四章代码保留教程的核心流程，并做了两处工程化补充：

1. **ReAct.py** — 教程在解析出无效 Action 格式时留了一个 `# ... 处理无效Action格式 ...` 的省略号。这里补全为：把格式错误作为 `Observation` 追加进历史，让模型下一轮自我纠正（对应 4.2.4 节「工具选择失败的处理」思路）。
2. **Plan_and_solve.py** — `Executor.execute()` 中显式初始化 `response_text = ""`，避免 `plan` 为空时循环变量未定义报 `UnboundLocalError`。

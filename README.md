# MyAgent：Hello-Agents 学习与本地适配

跟随 [Datawhale《Hello-Agents》](https://hello-agents.datawhale.cc/) 学习 Agent 范式、工具与记忆系统。代码按功能放在 `my_agent/`，手动运行示例按章节放在 `examples/`。

## 项目结构

```text
MyAgent/
├── my_agent/                 # 可复用实现
│   ├── core/                 # 消息、配置、Agent 抽象基类（7.3）
│   ├── llm/                  # 模型连接、Provider 识别和 ModelScope 扩展
│   ├── agents/               # 五种 Agent 的本地适配（7.4）
│   ├── tools/                # 计算、搜索、工具链和并行执行（7.5）
│   ├── memory/               # 记忆管理器、MemoryTool 和数据库适配（8.1–8.2）
│   └── paths.py              # 项目根目录、.env 和数据路径
├── examples/                 # 手动运行入口
│   ├── chapter04/            # ReAct、Plan-and-Solve、Reflection 等
│   ├── chapter07/            # 模型、对话、计算和搜索工具示例
│   └── chapter08/            # 记忆系统示例
├── docs/
│   └── project-layout.md     # 目录职责与旧文件迁移对照
├── data/                     # 本地记忆数据，运行时生成，已忽略
├── .env                      # 本地凭据，已忽略
├── .env.example              # 配置模板
├── pyproject.toml            # 项目依赖
└── uv.lock                   # 锁定依赖版本
```

目录职责和旧文件到新文件的对应关系见 [项目目录说明](docs/project-layout.md)。所有命令均在项目根目录执行，使用 `python -m` 按包路径运行示例。

## 安装与配置

需要 Python 3.10 及以上版本和 `uv`。保留已有 `.env`，首次使用时从模板创建：

```bash
test -f .env || cp .env.example .env
uv sync
```

使用 Qdrant/Neo4j 适配时安装可选依赖：

```bash
uv sync --extra memory-cloud
```

模型、搜索和数据库连接信息放在根目录 `.env`。智谱示例：

```dotenv
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=你的密钥
LLM_MODEL_ID=你的模型ID
```

模型提供商识别根据显式配置、环境变量和服务地址决定连接哪个服务，具体模型仍由 `LLM_MODEL_ID` 或代码参数指定。`my_agent/llm/client.py` 保留原有 `think(messages)` 接口和旧参数名；Hello-Agents SDK 的 Agent 使用 SDK 提供的 `HelloAgentsLLM.invoke()`，两者接口有所不同。

| 示例 | 外部服务 |
| --- | --- |
| 计算器、并行计算 | 不需要 |
| 普通对话、工具增强对话、Planner 实验 | 配置可用模型 |
| 第 4 章搜索 ReAct、搜索摘要工具链 | 配置 SerpApi 或 Tavily；ReAct 还需要模型 |
| 本机 Ollama | 本机启动 Ollama 并下载对应模型 |
| 记忆系统 | SQLite；配置云端连接后还会使用 Qdrant/Neo4j |

## 第 4 章：经典范式

```bash
uv run python -m examples.chapter04.react
uv run python -m examples.chapter04.plan_and_solve
uv run python -m examples.chapter04.reflection
uv run python -m examples.chapter04.planner_comparison
```

这些示例会调用模型。搜索版 ReAct 还使用 `.env` 中的 `SERPAPI_API_KEY`。第 4 章的 `ToolExecutor` 放在 `my_agent/tools/legacy.py`；第 7 章继续复用锁定版本 `hello-agents==0.1.1` 的 `ToolRegistry`。

## 第 7 章：模型、Agent 和工具

普通 SDK 对话，以及本地 `MySimpleAgent` 的对话、工具调用、流式响应示例：

```bash
uv run python -m examples.chapter07.basic_agent
uv run python -m examples.chapter07.simple_agent
```

第二个示例中的计算问题是 `15 * 8 + 32`，使用 SDK 的 `CalculatorTool`。原生 OpenAI tools 调用由 `my_agent/agents/function_call.py` 实现，服务端和所选模型需要支持 `tools` 参数。

本机 Ollama 示例：

```bash
ollama pull llama3:8b
uv run python -m examples.chapter07.local_model "请用中文简单介绍一下自己"
```

自定义计算器使用受限 AST 解释器，不执行 `eval`；并行执行器使用有界线程池，按输入顺序返回结果：

```bash
uv run python -m examples.chapter07.calculator
uv run python -m examples.chapter07.parallel_tools
```

计算器预期包含 `2 + 3 * 4 = 14`；并行示例预期返回 `['152', '10.0']`。在代码中复用：

```python
from my_agent.agents.simple import MySimpleAgent
from my_agent.tools.calculator import create_calculator_registry
from my_agent.tools.async_executor import AsyncToolExecutor
```

配置搜索密钥后运行搜索摘要工具链：

```bash
uv run python -m examples.chapter07.search_digest "Python 3.14 新特性"
```

它执行“网页搜索 → 筛选结果 → 格式化摘要”，保留来源和链接，不额外调用模型。SerpApi 使用基础依赖；Tavily 按需安装：

```bash
uv run --with tavily-python python -m examples.chapter07.search_digest "Python 3.14 新特性" --backend tavily
```

同时查询两个已配置来源时使用 `--backend hybrid`，并配置 `SERPAPI_API_KEY`、`TAVILY_API_KEY`。不使用的密钥请留空，避免把模板占位符当成有效配置。

## 第 8 章：记忆系统（8.1–8.2）

`my_agent/memory/manager.py` 提供 `MemoryItem`、`MemoryConfig`、`MemoryManager` 和 `MemoryTool`，支持四类记忆、按用户隔离、增删查改、三种遗忘策略和工作记忆整合。

- 工作记忆保存在内存中，按 TTL 和容量清理。
- 情景、语义和感知记忆默认保存在 SQLite。
- 工作记忆和感知记忆的文本描述使用本地词项匹配。
- 情景、语义记忆使用预训练 Embedding；情景记忆加入时间近因性，语义记忆结合图匹配分数。

运行章节示例：

```bash
uv run --extra memory-local --extra memory-cloud python -m examples.chapter08.memory_demo
```

示例中的 SQLite 使用内存数据库；已配置的 Qdrant/Neo4j 后端仍可能被调用。实际应用可以注册持久化的记忆工具：

```python
from hello_agents import ToolRegistry
from my_agent.memory import MemoryConfig, MemoryTool

memory = MemoryTool(
    user_id="user123",
    memory_config=MemoryConfig(database_path="data/memory.sqlite3"),
)
registry = ToolRegistry()
registry.register_tool(memory)

print(memory.run({
    "action": "add",
    "content": "用户正在学习 Python 和机器学习",
    "memory_type": "semantic",
    "importance": 0.8,
}))
print(memory.run({"action": "search", "query": "Python", "limit": 5}))
```

可用操作：`add`、`search`、`summary`、`stats`、`update`、`remove`、`forget`、`consolidate`、`clear_all`。`consolidate` 将达到重要性阈值的工作记忆转为情景记忆，并保存来源 ID。

`.env` 中的 `MEMORY_DB_PATH` 和显式传入的数据库相对路径均以项目根目录为基准；默认路径是 `data/memory.sqlite3`。配置加载位置固定为项目根目录 `.env`。

### 文本嵌入模型与向量检索

情景记忆和语义记忆使用真实的 Embedding 模型，不再使用字符哈希向量。两种运行方式任选其一；配置独立于对话模型的 `LLM_*`。

**本地开源模型（不需要 API Key）**

```bash
uv sync --extra memory-local --extra memory-cloud
```

在项目根目录 `.env` 中配置：

```dotenv
EMBED_MODEL_TYPE=local
EMBED_MODEL_NAME=BAAI/bge-small-zh-v1.5
EMBED_DIMENSIONS=
EMBED_DEVICE=cpu
```

本地方案使用 FastEmbed/ONNX 在 CPU 上运行 BGE，无需安装 PyTorch。首次使用时下载模型到被 Git 忽略的 `data/models/`，后续复用缓存。BGE small 中文模型输出 512 维向量；查询按模型建议添加检索指令，记忆原文不添加。它不收取 API 调用费，但需要本机资源和首次模型下载。模型的输入长度上限为 512 个 Token，超出部分由模型截断。其他 [FastEmbed 支持的文本模型](https://qdrant.github.io/fastembed/examples/Supported_Models/)可以通过 `EMBED_MODEL_NAME` 切换；模型变更时使用独立索引。

**阿里云百炼（云端 API）**

```bash
uv sync --extra memory-cloud
```

```dotenv
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v4
EMBED_DIMENSIONS=1024
EMBED_API_KEY=你的百炼APIKey
EMBED_BASE_URL=https://你的业务空间ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

也可用 `DASHSCOPE_API_KEY` 提供密钥；`EMBED_BASE_URL` 应使用控制台为实际业务空间提供的 OpenAI 兼容地址。若账户其他向量模型有可用额度，可将 `EMBED_MODEL_NAME` 改为对应模型（例如 `qwen3.7-text-embedding-flash`）。北京地域部分模型赠送有有效期的免费 Token，实际剩余量以控制台为准；免费额度不等于永久免费，使用前可在控制台开启目标模型的“免费额度用完即停”。参见[官方模型与额度说明](https://help.aliyun.com/zh/model-studio/embedding)和[BGE 模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)。

`my_agent/memory/backends.py` 提供 Qdrant/Neo4j 适配。设置 `QDRANT_URL`/`QDRANT_API_KEY` 后，向量由 Qdrant 保存、检索；未配置或连接失败时，使用同一个 Embedding 模型，在 SQLite 缓存向量并计算余弦相似度。不会悄悄替换为哈希向量，也不会在云端模型失败时自动改用另一个模型。工作记忆仍采用本地词项匹配。

Qdrant 实际集合名为 `QDRANT_COLLECTION_模型配置指纹`；模型、维度、服务地址或查询指令变更后使用独立索引，保留旧集合和 SQLite 原文。检索前自动为当前用户的候选记忆补建缺失或已过期的向量；也可以显式执行：

```bash
uv run --extra memory-local --extra memory-cloud python -m examples.chapter08.reindex_memory --user-id user123
```

百炼方案可省略 `--extra memory-local`。`--force` 会重新编码所有符合条件的记忆并消耗相应模型资源，正常补建会跳过已存在且内容未变化的向量。向量分数来自 Qdrant，SQLite 后端则计算归一化向量点积；管理器再融合时间、图匹配和重要性分数。

设置 `NEO4J_URI`/`NEO4J_PASSWORD` 后启用关系图。**目前实体和关系仍需通过元数据显式传入，尚未实现教程中的自动实体关系提取或多跳推理**：

```python
memory.run({
    "action": "add",
    "content": "Python 常用于机器学习",
    "memory_type": "semantic",
    "metadata": {
        "entities": ["Python", "机器学习"],
        "relations": [{"source": "Python", "type": "USED_FOR", "target": "机器学习"}],
    },
})
```

当前感知记忆管理文本描述和文件元数据，未执行图像/音频编码；第 8.3 节 RAG、MQE/HyDE 和第 8.4 节 PDF 问答助手尚未实现。此项目是章节练习代码。

## 本地测试

```bash
uv run --extra memory-cloud --extra memory-local python -m unittest discover -s tests -v
```

包含真实 BGE/ONNX 的离线集成测试，首次执行前需要下载模型。安装、模型准备和各项测试的覆盖范围见 [本地测试说明](docs/testing.md)。测试隔离真实 `.env`，不会访问云端账户。

## 提交与同步

```bash
git status
git add 具体文件或目录
git commit -m "说明本次修改"
git push
```

`git commit` 生成本地提交，`git push` 上传已有提交；未提交的文件修改不会随推送上传。可以比较以下提交号确认同步：

```bash
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

GitHub 文件列表的相对时间代表最后一次修改对应的提交时间；推送已有提交不会更新该时间。

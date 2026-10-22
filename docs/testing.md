# 本地测试

测试使用 Python 标准库 `unittest`。测试不读取项目真实 `.env`，不会调用真实百炼、Qdrant Cloud 或 Neo4j Aura；数据库使用临时 SQLite、本地 Qdrant 或模拟驱动，测试结束后清理临时数据。

## 安装与准备

```bash
uv sync --extra memory-cloud --extra memory-local
```

真实模型集成测试需要先下载 BGE/ONNX 文件。这一步仅下载公开模型，不执行推理，不需要 API Key：

```bash
uv run --extra memory-cloud --extra memory-local hf download Qdrant/bge-small-zh-v1.5 --cache-dir data/models --include model_optimized.onnx --include '*.json' --include vocab.txt
```

`data/models/` 已加入 Git 忽略。正式执行集成测试时开启离线模式，只使用上述缓存；缺少模型文件时测试会失败并显示错误，不会静默跳过。

## 执行

全部测试，包括真实 BGE/ONNX 推理和本地数据库集成：

```bash
uv run --extra memory-cloud --extra memory-local python -m unittest discover -s tests -v
```

只执行不需要模型文件的单元测试和模块导入检查：

```bash
uv run --extra memory-cloud --extra memory-local python -m unittest tests.test_memory tests.test_embedding tests.test_backends tests.test_project_layout -v
```

## 覆盖范围

| 文件 | 覆盖内容 |
| --- | --- |
| `tests/test_memory.py` | 四类记忆已实现的操作、用户/会话隔离、TTL/容量、持久化、排序、遗忘/整合、向量缓存/补建、故障处理、MemoryTool |
| `tests/test_embedding.py` | 本地/百炼提供商选择、配置校验、请求分批、返回顺序、向量归一化、模型身份、查询指令与异常 |
| `tests/test_backends.py` | 本地 Qdrant 的过滤/分数/增删/幂等及集合隔离；Neo4j 模拟驱动的参数、用户隔离和关闭 |
| `tests/test_local_embedding_integration.py` | 真实 BGE 的 512 维归一化输出、语义相关性排序，以及 BGE → SQLite/Qdrant → MemoryManager 的完整本地链路 |
| `tests/test_project_layout.py` | 重构后的模块可以导入，导入时不会创建模型客户端或加载模型，配置路径固定到项目根目录 |

感知记忆目前仅管理文本描述和文件元数据，相关测试验证保存、检索、更新和删除；不代表已完成图像/音频编码或跨模态检索。Neo4j 自动实体关系提取尚未实现；百炼 API 和云数据库的真实可用性也不属于本地测试结论。

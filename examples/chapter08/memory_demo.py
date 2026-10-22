"""手动运行示例：chapter08/memory_demo.py."""

from my_agent.memory import MemoryConfig, MemoryTool


def main() -> None:
    tool = MemoryTool(user_id="chapter8-demo", memory_config=MemoryConfig(database_path=":memory:"))
    print(tool.run({"action": "add", "content": "张三正在学习 Python 和记忆系统", "memory_type": "semantic", "importance": 0.9}))
    print(tool.run({"action": "add", "content": "刚刚讨论了中文分词和向量检索", "memory_type": "episodic", "importance": 0.7}))
    print(tool.run({"action": "search", "query": "Python 学习进度"}))
    print(tool.run({"action": "summary"}))


if __name__ == "__main__":
    main()

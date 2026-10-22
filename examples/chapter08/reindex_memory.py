"""为已有 SQLite 记忆补建当前模型的向量索引，不生成测试记忆。"""

import argparse

from my_agent.memory import MemoryConfig, MemoryManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True, help="需要补建索引的用户 ID")
    parser.add_argument("--database", help="SQLite 文件路径；默认读取 .env 的 MEMORY_DB_PATH")
    parser.add_argument("--force", action="store_true", help="即使内容未变化，也重新调用模型编码")
    args = parser.parse_args()
    config = MemoryConfig(database_path=args.database) if args.database else None
    manager = MemoryManager(user_id=args.user_id, config=config)
    count = manager.reindex_memories(force=args.force)
    print(f"已补建 {count} 条情景/语义记忆向量，模型：{manager.embedder.model_name}")


if __name__ == "__main__":
    main()

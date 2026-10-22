"""第 7.5.4 节：搜索、筛选和格式化摘要，需要已配置的搜索服务。"""

import argparse

from dotenv import load_dotenv

from my_agent.paths import ENV_FILE
from my_agent.tools.chain import ToolChainManager, create_search_digest_chain
from my_agent.tools.search import create_advanced_search_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="Python 3.14 新特性")
    parser.add_argument("--backend", choices=("serpapi", "tavily", "hybrid"), default="serpapi")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    registry = create_advanced_search_registry(backend=args.backend)
    chains = ToolChainManager(registry)
    chains.register_chain(create_search_digest_chain())
    print(chains.execute_chain("search_digest", args.query))


if __name__ == "__main__":
    main()

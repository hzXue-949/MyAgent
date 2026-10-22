"""第 7.5.4 节：并行运行两个本地计算任务。"""

import asyncio

from my_agent.tools.async_executor import AsyncToolExecutor
from my_agent.tools.calculator import create_calculator_registry


async def main() -> None:
    registry = create_calculator_registry()
    with AsyncToolExecutor(registry, max_workers=2) as executor:
        results = await executor.execute_tools_parallel([
            {"tool_name": "my_calculator", "input_data": "15 * 8 + 32"},
            {"tool_name": "my_calculator", "input_data": "sqrt(16) + 2 * 3"},
        ])
    print(results)


if __name__ == "__main__":
    asyncio.run(main())

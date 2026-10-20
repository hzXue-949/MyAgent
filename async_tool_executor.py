"""第 7.5.4 节：在线程池中并行运行可独立执行的工具。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from hello_agents import ToolRegistry


class ToolTask(TypedDict):
    tool_name: str
    input_data: str


class AsyncToolExecutor:
    """把同步注册工具提交到有界线程池，并按输入顺序返回结果。"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ValueError("max_workers 必须大于 0")
        self.registry = registry
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="agent-tool"
        )
        self._closed = False

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        if self._closed:
            raise RuntimeError("异步工具执行器已关闭")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.registry.execute_tool, tool_name, input_data
        )

    async def execute_tools_parallel(self, tasks: list[ToolTask]) -> list[str]:
        """并发执行多个互不依赖的工具任务，结果顺序与 tasks 一致。"""
        for index, task in enumerate(tasks):
            if not task.get("tool_name"):
                raise ValueError(f"第 {index + 1} 个任务缺少 tool_name")
            if not isinstance(task.get("input_data", ""), str):
                raise TypeError(f"第 {index + 1} 个任务的 input_data 必须是字符串")
        return list(
            await asyncio.gather(
                *(
                    self.execute_tool_async(task["tool_name"], task.get("input_data", ""))
                    for task in tasks
                )
            )
        )

    async def execute_tools_batch(
        self, tool_name: str, input_list: list[str]
    ) -> list[str]:
        """并行执行同一个工具处理的一组输入。"""
        return await self.execute_tools_parallel(
            [{"tool_name": tool_name, "input_data": item} for item in input_list]
        )

    def close(self) -> None:
        if not self._closed:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True

    def __enter__(self) -> "AsyncToolExecutor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


async def run_parallel_tools(
    registry: ToolRegistry, tasks: list[ToolTask], max_workers: int = 4
) -> list[str]:
    """以一次性执行器并行运行一组工具任务。"""
    with AsyncToolExecutor(registry, max_workers=max_workers) as executor:
        return await executor.execute_tools_parallel(tasks)

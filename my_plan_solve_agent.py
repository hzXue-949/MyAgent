"""第 7.4.4 节：提供结构化提示词的 Plan-and-Solve Agent。"""

import ast
import json
import re
from typing import Any, Dict, Optional

from hello_agents import Config, HelloAgentsLLM, Message, PlanAndSolveAgent


DEFAULT_PROMPTS: Dict[str, str] = {
    "planner": """你是一个规划专家。请将用户提出的问题分解为按逻辑顺序排列、可执行的步骤。

问题：{question}

请只输出 Python 字符串列表，不要添加其他解释：
```python
["步骤1", "步骤2", "步骤3"]
```""",
    "executor": """你是一个执行专家。请根据完整计划和已完成的步骤，解决当前步骤。
只输出当前步骤的结果，不要提前回答整个问题。

原始问题：
{question}

完整计划：
{plan}

已完成步骤及结果：
{history}

当前步骤：
{current_step}

请给出当前步骤的结果：""",
}


class MyPlanAndSolveAgent(PlanAndSolveAgent):
    """继承框架的规划和逐步执行流程，并允许逐项覆盖提示词。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        prompts = {**DEFAULT_PROMPTS, **(custom_prompts or {})}
        missing = set(DEFAULT_PROMPTS) - prompts.keys()
        if missing:
            raise ValueError(f"Plan-and-Solve 提示词缺少模板: {', '.join(sorted(missing))}")
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            custom_prompts=prompts,
        )
        self.last_plan: list[str] = []
        self.last_execution_history: list[str] = []

    def run(self, input_text: str, **kwargs: Any) -> str:
        """先生成步骤计划，再逐步执行，并保存最终问答。"""
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        plan_prompt = self.prompts["planner"].format(question=input_text)
        raw_plan = self._invoke_prompt(plan_prompt, **kwargs)
        plan = self._parse_plan(raw_plan)

        if not plan:
            final_answer = "无法从模型响应中解析出有效计划，任务终止。"
            self.add_message(Message(content=input_text, role="user"))
            self.add_message(Message(content=final_answer, role="assistant"))
            print(f"\n--- 任务终止 ---\n模型规划响应:\n{raw_plan}")
            return final_answer

        self.last_plan = plan
        history = []
        final_answer = ""
        for index, step in enumerate(plan, start=1):
            print(f"\n-> 正在执行步骤 {index}/{len(plan)}: {step}")
            execute_prompt = self.prompts["executor"].format(
                question=input_text,
                plan=plan,
                history="\n".join(history) or "无",
                current_step=step,
            )
            final_answer = self._invoke_prompt(execute_prompt, **kwargs)
            history.append(f"步骤 {index}: {step}\n结果: {final_answer}")

        self.last_execution_history = history
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_answer, role="assistant"))
        return final_answer

    def _invoke_prompt(self, prompt: str, **kwargs: Any) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.llm.invoke(messages, **kwargs) or ""

    @staticmethod
    def _parse_plan(response: str) -> list[str]:
        """安全解析 Python/JSON 字符串列表，不执行模型返回的代码。"""
        if not response or not response.strip():
            return []

        fenced = re.search(r"```(?:python|json)?\s*(.*?)```", response, re.IGNORECASE | re.DOTALL)
        candidate = fenced.group(1).strip() if fenced else response.strip()
        bracketed = re.search(r"\[[\s\S]*\]", candidate)
        if bracketed:
            candidate = bracketed.group(0)

        for parser in (ast.literal_eval, json.loads):
            try:
                value = parser(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(value, list):
                return [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return []

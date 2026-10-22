"""第 7.4.3 节：提供通用提示词的 ReflectionAgent。"""

from typing import Any, Dict, Optional

from hello_agents import Config, HelloAgentsLLM, Message, ReflectionAgent


DEFAULT_PROMPTS: Dict[str, str] = {
    "initial": """请根据以下要求完成任务：

任务：{task}

请提供一个完整、准确的回答。""",
    "reflect": """请仔细审查以下回答，找出可能的问题或改进空间。

原始任务：
{task}

当前回答：
{content}

请分析回答质量并提出具体改进建议。如果回答已经很好，请回答“无需改进”。""",
    "refine": """请根据反馈改进回答。

原始任务：
{task}

上一轮回答：
{last_attempt}

反馈意见：
{feedback}

请提供改进后的完整回答。""",
}


class MyReflectionAgent(ReflectionAgent):
    """继承框架的反思循环，并允许逐项覆盖默认提示词。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        prompts = {**DEFAULT_PROMPTS, **(custom_prompts or {})}
        missing = set(DEFAULT_PROMPTS) - prompts.keys()
        if missing:
            raise ValueError(f"Reflection 提示词缺少模板: {', '.join(sorted(missing))}")
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            max_iterations=max_iterations,
            custom_prompts=prompts,
        )

    def run(self, input_text: str, **kwargs: Any) -> str:
        """生成初稿，按反馈反思并迭代改写，最后保存本轮问答。"""
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        final_result = self._invoke_prompt(initial_prompt, **kwargs)

        for iteration in range(self.max_iterations):
            print(f"\n--- 第 {iteration + 1}/{self.max_iterations} 轮反思 ---")
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=final_result,
            )
            feedback = self._invoke_prompt(reflect_prompt, **kwargs)
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                break

            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=final_result,
                feedback=feedback,
            )
            final_result = self._invoke_prompt(refine_prompt, **kwargs)

        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_result, role="assistant"))
        return final_result

    def _invoke_prompt(self, prompt: str, **kwargs: Any) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.llm.invoke(messages, **kwargs) or ""

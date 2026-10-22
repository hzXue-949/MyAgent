"""
第四章 4.2：ReAct (Reasoning and Acting) 智能体

将"思考 (Thought) - 行动 (Action) - 观察 (Observation)"显式结合的经典范式，
让智能体边想边做、动态调整，能够调用外部工具回答知识库之外的问题。
"""

import re
from datetime import date

from my_agent.llm.client import HelloAgentsLLM
from my_agent.tools.legacy import ToolExecutor, search

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

**当前日期：{today}**

可用工具如下:
{tools}

请严格按照以下格式进行回应，**每次只输出一个 Thought 和一个 Action，然后停止**：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- tool_name[tool_input]：调用一个可用工具。
- Finish[最终答案]：当你认为已经获得最终答案时。

⚠️ 重要规则：
1. 每次回复只包含一个 Thought 和一个 Action，**不要**自己模拟 Result 或 Observation。
2. 工具调用后，系统会自动返回真实的 Observation，你再基于此输出下一轮 Thought + Action。
3. 当你收集到足够的信息时，使用 Finish[最终答案] 结束。
4. 你的训练知识可能已经过时，**回答时事类问题时必须以搜索到的 Observation 为准**，
   不要用记忆里的旧信息去覆盖搜索结果。
5. 搜索时**不要**在查询词里写具体年份。搜索引擎默认返回最新结果，
   写死年份反而会把你锁在过时信息里（例如该写 "iPhone 最新型号" 而不是
   "iPhone 最新型号 2025"）。
6. 判断信息新旧时，请看 Observation 里标注的发布日期；带日期的结果比不带日期的更可信。

现在，请开始解决以下问题:
Question: {question}
History:
{history}
"""


class ReActAgent:
    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int = 5,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = []  # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                today=date.today().isoformat(),
                tools=tools_desc,
                question=question,
                history=history_str,
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"🤔 思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                finish_match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if not finish_match:
                    # 模型没有给出合法的 Finish[...]，把格式错误反馈给它继续
                    observation = (
                        f"错误:Finish 格式不合法 '{action[:80]}'，"
                        "请严格使用 Finish[最终答案] 格式。"
                    )
                    print(f"⚠️  {observation}")
                    self.history.append(f"Observation: {observation}")
                    continue
                final_answer = finish_match.group(1).strip()
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # 处理无效的Action格式，将其作为观察反馈给LLM后继续
                observation = f"错误:无法解析的Action格式 '{action}'，请严格使用 tool_name[tool_input] 格式。"
                print(f"⚠️  {observation}")
                self.history.append(f"Observation: {observation}")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input)  # 调用真实工具

            # 5. 观测结果的整合
            print(f"👀 观察: {observation}")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """解析LLM的输出，提取第一个 Thought 和第一个 Action。

        这里要同时躲开两个坑：
        1. 模型有时会在一次回复里自问自答地生成多轮 Thought/Action/Result，
           所以只能取「第一个」Action，不能贪婪匹配到文本末尾。
        2. `Finish[最终答案]` 的答案常常跨多行，闭合的 `]` 在好几行之后，
           所以首行括号没闭合时要向后补齐。
        """
        # Thought: 匹配到第一个 Action: 或行尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None

        # 只取第一个 Action: 那一行
        line_match = re.search(r"^Action:\s*(.+)$", text, re.MULTILINE)
        if not line_match:
            return thought, None
        action = line_match.group(1).strip()

        # 首行括号未闭合（典型是多行的 Finish[...]），补齐到闭合的 ]
        if action.count("[") > action.count("]"):
            remainder = text[line_match.end():]
            close = remainder.rfind("]")
            if close != -1:
                action = action + remainder[: close + 1]

        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。

        兼容模型偶尔在 Action 两端加反引号的情况，例如 `Search[query]`。
        """
        # 去除两端可能的反引号
        cleaned = action_text.strip().strip("`")
        match = re.match(r"(\w+)\[(.*)\]", cleaned, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None


if __name__ == "__main__":
    # 1. 初始化LLM客户端与工具执行器
    llm_client = HelloAgentsLLM()
    tool_executor = ToolExecutor()

    # 2. 注册搜索工具
    search_description = (
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    )
    tool_executor.registerTool("Search", search_description, search)

    # 3. 运行ReAct智能体，回答一个时效性问题
    agent = ReActAgent(llm_client, tool_executor, max_steps=5)
    question = "苹果最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)

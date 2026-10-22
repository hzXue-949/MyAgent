"""
Planner 提示词 A/B 对比实验

目的：验证 Plan-and-Solve 中「计划阶段把答案算完」的成因与修复。
- A 组：教程原版 Planner 提示词（只要求分解步骤，未禁止计算）
- B 组：加了「禁止出现任何数值结果，只能用符号引用前序步骤」约束的提示词

只跑 Planner，不跑 Executor，便于直接观察计划内容是否泄漏答案。
"""

import ast
import re

from my_agent.llm.client import HelloAgentsLLM

# A 组：与 chapter04/plan_and_solve.py 的 Planner 提示词一致
PROMPT_A = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# B 组：显式禁止在计划中给出中间结果与最终答案
PROMPT_B = """
你是一个顶级的AI规划专家。你的任务是把用户问题分解成一份「待执行」的行动计划，交给另一个执行者去算。

硬性约束（违反即视为失败）：
1. 你只负责描述「要做什么操作」，绝对不能自己计算，也不能给出任何中间结果或最终答案。
2. 步骤文本中不允许出现任何由推理得出的数字；只有直接抄自题干的已知条件可以出现数字。
3. 需要用到前面步骤的产物时，必须用「步骤N的结果」这种符号方式引用，不得写出它的具体数值。
4. 每个步骤是一个独立可执行的子任务，按逻辑顺序排列。

反例（禁止）："计算周二的数量：15 × 2 = 30 个"
正例（允许）："根据题干「周二是周一的两倍」，用步骤1的结果计算周二卖出的数量"

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

QUESTION = (
    "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。"
    "周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
)

# 题干里本来就有的已知数字，出现这些不算泄漏
KNOWN_NUMBERS = {"15", "5", "2", "3"}
LEAK_NUMBERS = {"30", "25", "70"}


def parse_plan(response_text: str) -> list[str]:
    try:
        plan_str = response_text.split("```python")[1].split("```")[0].strip()
        plan = ast.literal_eval(plan_str)
        return plan if isinstance(plan, list) else []
    except Exception as e:  # noqa: BLE001
        print(f"❌ 解析失败: {e}\n原始响应: {response_text}")
        return []


def check_leak(plan: list[str]) -> None:
    leaked = []
    for i, step in enumerate(plan, 1):
        nums = set(re.findall(r"\d+", step))
        hit = nums & LEAK_NUMBERS
        if hit:
            leaked.append((i, sorted(hit), step))
    if leaked:
        print(f"🚨 检测到答案泄漏（{len(leaked)}/{len(plan)} 个步骤含推导出的数值）:")
        for i, hit, step in leaked:
            print(f"   - 步骤{i} 含 {hit}: {step}")
    else:
        print(f"✅ 无答案泄漏，{len(plan)} 个步骤均为纯操作描述")


def run(label: str, template: str) -> None:
    print(f"\n{'=' * 70}\n【{label}】\n{'=' * 70}")
    llm = HelloAgentsLLM()
    resp = llm.think(messages=[{"role": "user", "content": template.format(question=QUESTION)}]) or ""
    plan = parse_plan(resp)
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")
    print()
    check_leak(plan)


if __name__ == "__main__":
    run("A 组 · 教程原版提示词", PROMPT_A)
    run("B 组 · 加了禁止计算约束", PROMPT_B)

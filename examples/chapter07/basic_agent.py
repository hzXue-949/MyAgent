"""第 7 章手动对话示例；直接运行时会请求配置的模型服务。"""

from hello_agents import SimpleAgent, HelloAgentsLLM
from dotenv import load_dotenv
from hello_agents.tools import CalculatorTool

from my_agent.paths import ENV_FILE


def main() -> None:
    # 在项目根目录 .env 配置模型服务。

    # 加载环境变量
    load_dotenv(ENV_FILE)

    # 创建LLM实例 - 框架自动检测provider
    llm = HelloAgentsLLM()

    # 或手动指定provider（可选）
    # llm = HelloAgentsLLM(provider="modelscope")

    # 创建SimpleAgent
    agent = SimpleAgent(
        name="AI助手",
        llm=llm,
        system_prompt="你是一个有用的AI助手"
    )

    # 基础对话
    response = agent.run("你好！请介绍一下自己")
    print(response)

    # 添加工具功能（可选）
    calculator = CalculatorTool()
    # 需要实现7.4.1的MySimpleAgent进行调用，后续章节会支持此类调用方式
    # agent.add_tool(calculator)

    # 现在可以使用工具了
    response = agent.run("请帮我计算 2 + 3 * 4")
    print(response)

    # 查看对话历史
    print(f"历史消息数: {len(agent.get_history())}")


if __name__ == "__main__":
    main()

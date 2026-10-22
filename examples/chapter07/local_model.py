"""第 7.2.2 节：通过 HelloAgentsLLM 调用本机 Ollama 模型。"""

import argparse

from hello_agents import HelloAgentsLLM
from hello_agents.core.exceptions import HelloAgentsException


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="请用中文简单介绍一下你自己。")
    parser.add_argument("--model", default="llama3:8b", help="已下载的 Ollama 模型名称")
    args = parser.parse_args()

    # 显式指定本机配置，不读取或修改项目现有的云端模型配置。
    llm = HelloAgentsLLM(
        provider="ollama",
        model=args.model,
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",  # 本地接口不需要真实密钥，SDK 要求非空占位值。
        temperature=0.2,
        max_tokens=512,
        timeout=120,
    )
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手，请用简体中文简洁回答。"},
        {"role": "user", "content": args.prompt},
    ]
    try:
        # think() 内部已逐字打印；消费生成器即可，不要重复 print。
        response = "".join(llm.think(messages))
        if not response.strip():
            raise SystemExit("模型返回了空响应，请检查 Ollama 服务日志。")
    except HelloAgentsException as exc:
        raise SystemExit(
            f"本地模型调用失败：{exc}\n"
            f"请确认 Ollama 服务已启动，并已执行 ollama pull {args.model}。"
        ) from exc


if __name__ == "__main__":
    main()

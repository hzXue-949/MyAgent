"""第 7.4.5 节：使用 OpenAI 兼容原生工具调用的 Agent。"""

import inspect
import json
from typing import Any, Dict, Optional, get_args, get_origin

from hello_agents import Config, HelloAgentsLLM, Message, ToolRegistry
from hello_agents.core.agent import Agent


def _json_type(value: Any) -> str:
    """将 Python 类型或 ToolParameter.type 转成 JSON Schema 类型。"""
    if isinstance(value, str):
        normalized = value.lower()
        return {
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "dict": "object",
            "object": "object",
            "list": "array",
            "array": "array",
        }.get(normalized, "string")

    origin = get_origin(value)
    if origin is not None:
        args = [arg for arg in get_args(value) if arg is not type(None)]
        if origin in {list, tuple, set}:
            return "array"
        if origin is dict:
            return "object"
        if args:
            return _json_type(args[0])

    if value is int:
        return "integer"
    if value is float:
        return "number"
    if value is bool:
        return "boolean"
    if value is dict:
        return "object"
    if value in {list, tuple, set}:
        return "array"
    return "string"


class MyFunctionCallAgent(Agent):
    """把 ToolRegistry 工具转成 API schema，并执行模型返回的工具调用。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_tool_iterations: int = 5,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        if max_tool_iterations < 1:
            raise ValueError("max_tool_iterations 必须大于 0")
        if not tool_registry or not tool_registry.list_tools():
            raise ValueError("FunctionCallAgent 至少需要注册一个工具")
        self.tool_registry = tool_registry
        self.max_tool_iterations = max_tool_iterations

    def run(
        self,
        input_text: str,
        max_tool_iterations: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """运行对话；在工具执行完成后继续请求模型组织最终回答。"""
        client = getattr(self.llm, "_client", None) or getattr(self.llm, "client", None)
        if client is None:
            raise RuntimeError("LLM 客户端没有可用的 OpenAI 兼容 client")

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(message.to_dict() for message in self._history)
        messages.append({"role": "user", "content": input_text})

        tools = self._build_tool_schemas()
        max_iterations = (
            self.max_tool_iterations
            if max_tool_iterations is None
            else max_tool_iterations
        )
        if max_iterations < 1:
            raise ValueError("max_tool_iterations 必须大于 0")

        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("temperature", self.llm.temperature)
        max_tokens = getattr(self.llm, "max_tokens", None)
        if max_tokens is not None:
            request_kwargs.setdefault("max_tokens", max_tokens)

        tool_rounds = 0
        final_response = ""
        while True:
            call_kwargs = {
                "model": self.llm.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                **request_kwargs,
            }

            completion = client.chat.completions.create(**call_kwargs)
            assistant_message = completion.choices[0].message
            tool_calls = assistant_message.tool_calls or []

            if not tool_calls:
                final_response = assistant_message.content or ""
                break
            if tool_rounds >= max_iterations:
                final_response = "工具调用轮数已达到上限，未能生成最终回答。"
                break

            messages.append(assistant_message.model_dump(exclude_none=True))
            for tool_call in tool_calls:
                result = self._run_tool_call(tool_call.function.name, tool_call.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            tool_rounds += 1

        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_response, role="assistant"))
        return final_response

    def _build_tool_schemas(self) -> list[Dict[str, Any]]:
        """根据 ToolParameter 和注册函数签名生成 OpenAI function schema。"""
        schemas = []
        descriptions = {}
        for line in self.tool_registry.get_tools_description().splitlines():
            if line.startswith("- ") and ":" in line:
                name, description = line[2:].split(":", 1)
                descriptions[name.strip()] = description.strip()

        for name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(name)
            if tool is not None:
                properties = {}
                required = []
                for parameter in tool.get_parameters():
                    schema: Dict[str, Any] = {
                        "type": _json_type(parameter.type),
                        "description": parameter.description,
                    }
                    if parameter.default is not None:
                        schema["default"] = parameter.default
                    properties[parameter.name] = schema
                    if parameter.required:
                        required.append(parameter.name)
            else:
                function = self.tool_registry.get_function(name)
                if function is None:
                    continue
                properties, required = self._function_parameters(function)

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": descriptions.get(name, name),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return schemas

    @staticmethod
    def _function_parameters(function: Any) -> tuple[Dict[str, Any], list[str]]:
        properties = {}
        required = []
        for name, parameter in inspect.signature(function).parameters.items():
            if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
                continue
            annotation = parameter.annotation
            properties[name] = {
                "type": _json_type(annotation),
                "description": "工具函数参数",
            }
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
        if not properties:
            properties["input"] = {"type": "string", "description": "工具输入"}
            required.append("input")
        return properties, required

    def _run_tool_call(self, name: str, arguments_json: str) -> str:
        """解析参数、执行已注册工具，并把结果转为对话文本。"""
        try:
            arguments = json.loads(arguments_json or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")

            tool = self.tool_registry.get_tool(name)
            if tool is not None:
                if not tool.validate_parameters(arguments):
                    return f"工具 {name} 参数缺失或无效：{arguments}"
                result = tool.run(arguments)
            else:
                function = self.tool_registry.get_function(name)
                if function is None:
                    return f"未找到工具：{name}"
                signature = inspect.signature(function)
                parameters = list(signature.parameters.values())
                if len(parameters) == 1 and parameters[0].name not in arguments:
                    result = function(json.dumps(arguments, ensure_ascii=False))
                else:
                    result = function(**arguments)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return f"工具 {name} 执行失败：{exc}"

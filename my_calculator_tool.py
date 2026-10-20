"""第 7.5.2 节：注册一个只计算安全数学表达式的工具。"""

import ast
import math
import operator
from typing import Any, Callable

from hello_agents import ToolRegistry


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "sqrt": math.sqrt,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_MAX_EXPRESSION_LENGTH = 200
_MAX_AST_NODES = 64
_MAX_POWER_EXPONENT = 12
_MAX_RESULT_MAGNITUDE = 1e100


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER_EXPONENT:
            raise ValueError("乘方指数绝对值不能超过 12")
        value = _BINARY_OPERATORS[type(node.op)](left, right)
        return _validate_number(value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _validate_number(_UNARY_OPERATORS[type(node.op)](_evaluate(node.operand)))

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        args = [_evaluate(arg) for arg in node.args]
        return _validate_number(_FUNCTIONS[node.func.id](*args))

    raise ValueError(f"不支持的表达式部分：{type(node).__name__}")


def _validate_number(value: Any) -> int | float:
    if type(value) not in (int, float):
        raise ValueError("表达式结果必须是数字")
    if abs(value) > _MAX_RESULT_MAGNITUDE:
        raise ValueError("表达式结果超出允许范围")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("表达式结果必须是有限数值")
    return value


def my_calculate(expression: str) -> str:
    """计算基本四则运算、乘方、sqrt/abs、pi 和 e，不执行任意 Python 代码。"""
    if not isinstance(expression, str) or not expression.strip():
        return "计算表达式不能为空"
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        return f"计算表达式不能超过 {_MAX_EXPRESSION_LENGTH} 个字符"

    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise ValueError("计算表达式过于复杂")
        return str(_evaluate(tree.body))
    except (ArithmeticError, SyntaxError, TypeError, ValueError) as exc:
        return f"计算失败：{exc}"


def register_calculator(registry: ToolRegistry) -> ToolRegistry:
    """将计算器函数注册到现有注册表，便于和其他工具组合。"""
    registry.register_function(
        name="my_calculator",
        description="安全计算数学表达式，支持 +、-、*、/、乘方、sqrt、abs、pi 和 e。",
        func=my_calculate,
    )
    return registry


def create_calculator_registry() -> ToolRegistry:
    """创建只包含计算器的工具注册表。"""
    return register_calculator(ToolRegistry())


if __name__ == "__main__":
    registry = create_calculator_registry()
    for expression in ("2 + 3 * 4", "sqrt(16) + pi"):
        print(f"{expression} = {registry.execute_tool('my_calculator', expression)}")

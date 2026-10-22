"""手动运行示例：chapter07/calculator.py."""

from my_agent.tools.calculator import create_calculator_registry


def main() -> None:
    registry = create_calculator_registry()
    for expression in ("2 + 3 * 4", "sqrt(16) + pi"):
        print(f"{expression} = {registry.execute_tool('my_calculator', expression)}")


if __name__ == "__main__":
    main()

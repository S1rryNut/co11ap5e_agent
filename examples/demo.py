"""
ReAct Agent 完整 Demo。

运行：uv run python examples/demo.py
"""
import os
import asyncio
from dotenv import load_dotenv
from react_agent.llm import LLMClient
from react_agent.agent import Agent
from react_agent.tool import tool


# ========== 示例工具 ==========

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除和括号，比如 '123 * 456' 或 '(35 - 8) * 2 + 100'。"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算出错: {e}"


@tool
def get_weather(city: str) -> str:
    """查询某城市的当前天气。"""
    data = {
        "北京": "晴，25℃，风速 3 级",
        "上海": "多云，28℃，湿度 65%",
        "苏州": "小雨，24℃，湿度 80%",
        "深圳": "雷阵雨，30℃，湿度 90%",
    }
    return data.get(city, f"暂无 {city} 的天气数据")


@tool
def get_time() -> str:
    """获取当前时间。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ========== 主程序 ==========

async def main():
    load_dotenv()

    llm = LLMClient(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    agent = Agent(
        llm=llm,
        tools=[calculator, get_weather, get_time],
        system_prompt="你是一个智能助手。需要计算用 calculator，需要天气用 get_weather，需要时间用 get_time。",
        max_iterations=10,
        verbose=True,
    )

    print("=" * 50)
    print("ReAct Agent Demo（输入 exit 退出）")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break

        try:
            answer = await agent.arun(user_input)
            print(f"\n助手: {answer}")
        except Exception as e:
            print(f"\n出错了: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())

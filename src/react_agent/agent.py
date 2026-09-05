"""
ReAct Agent 核心循环。

思考 → 调工具 → 看结果 → 再思考，直到完成。
"""
import asyncio
from react_agent.llm import LLMClient
from react_agent.tool import Tool, ToolRegistry
from react_agent.memory.base import BaseMemory
from react_agent.memory.short_term import ShortTermMemory
from react_agent.errors import (
    MaxIterationsError,
    ConsecutiveFailureError,
)


class Agent:
    """ReAct Agent：思考 → 调工具 → 看结果 → 再思考，直到完成。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        memory: BaseMemory | None = None,
        system_prompt: str = "",
        max_iterations: int = 10,
        verbose: bool = False,
        callbacks: dict | None = None,
    ):
        self.llm = llm

        # 工具注册到 ToolRegistry
        self.registry = ToolRegistry()
        for t in tools:
            self.registry.register(t)

        # 记忆：没传就默认用 ShortTermMemory
        self.memory = memory or ShortTermMemory(system_prompt=system_prompt)
        self.system_prompt = system_prompt

        # 最多循环 10 轮，防止死循环
        self.max_iterations = max_iterations

        # verbose=True 时打印每轮过程
        self.verbose = verbose

        # 回调函数
        self.callbacks = callbacks or {}

        # 连续失败计数：{工具名: 连续失败次数}
        self._failure_count: dict[str, int] = {}

    async def arun(self, user_input: str) -> str:
        """异步运行 Agent，返回最终回答。"""
        # 第 1 步：用户消息加入记忆
        self.memory.add({"role": "user", "content": user_input})

        # 第 2 步：ReAct 循环
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- 第 {iteration + 1} 轮 ---")

            if "on_iteration_start" in self.callbacks:
                self.callbacks["on_iteration_start"](iteration)

            # 2.1 调 LLM
            messages = self.memory.get_messages()
            response = await self.llm.achat(
                messages,
                tools=self.registry.get_schemas(),
            )

            # 2.2 没有 tool_calls = 最终回答
            if not response["tool_calls"]:
                answer = response["content"]
                self.memory.add({"role": "assistant", "content": answer})
                if self.verbose:
                    print(f"最终回答: {answer}")
                return answer

            # 2.3 有 tool_calls，先把 assistant 消息加入记忆
            self.memory.add({
                "role": "assistant",
                "content": response["content"],
                "tool_calls": response["tool_calls"],
            })

            if self.verbose:
                for tc in response["tool_calls"]:
                    print(f"调用工具: {tc['function']['name']}({tc['function']['arguments'][:60]})")

            # 2.4 并行执行所有工具
            tool_tasks = [
                self._execute_tool(tc) for tc in response["tool_calls"]
            ]
            tool_results = await asyncio.gather(*tool_tasks)

            # 2.5 工具结果加入记忆
            for tool_msg in tool_results:
                self.memory.add(tool_msg)
                if self.verbose:
                    print(f"工具结果: {tool_msg['content'][:80]}")

        # 第 3 步：达到最大迭代次数
        raise MaxIterationsError(f"达到最大迭代次数 {self.max_iterations}")

    async def _execute_tool(self, tool_call: dict) -> dict:
        """执行一个工具调用，返回 tool 格式的消息。"""
        name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        tool_call_id = tool_call["id"]

        if "on_tool_call" in self.callbacks:
            self.callbacks["on_tool_call"](name, arguments)

        tool_obj = self.registry.get(name)

        if tool_obj is None:
            result = f"工具 '{name}' 不存在，可用工具：{self.registry.names()}"
            self._record_failure(name)
        else:
            result = await tool_obj.ainvoke(arguments)

            # 判断是否失败
            if any(kw in result for kw in ["出错", "失败", "超时", "解析失败", "不存在"]):
                self._record_failure(name)
            else:
                self._failure_count[name] = 0

        if "on_tool_result" in self.callbacks:
            self.callbacks["on_tool_result"](tool_call_id, result)

        return {
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call_id,
        }

    def _record_failure(self, tool_name: str):
        """记录工具连续失败，超过 3 次抛异常。"""
        self._failure_count[tool_name] = self._failure_count.get(tool_name, 0) + 1
        if self._failure_count[tool_name] >= 3:
            raise ConsecutiveFailureError(
                f"工具 '{tool_name}' 连续失败 {self._failure_count[tool_name]} 次"
            )

    def run(self, user_input: str) -> str:
        """同步包装。"""
        return asyncio.run(self.arun(user_input))

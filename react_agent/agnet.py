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
    # 

    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        memory: BaseMemory | None = None,
        max_iterations: int = 10,
        system_prompt: str | None = None,
        verbose: bool = False,
        callback: callable | None = None,
    ):
        self.llm = llm
        self.registry = ToolRegistry()
        for tool in tools:
            self.registry.register(tool)
        self.memory = memory or ShortTermMemory(system_prompt=system_prompt)
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.callback = callback or {}
        self._failure_count = dict[str, int]  # 连续失败计数器

    async def arun(self, user_input: str) -> str:
        # 1. 用户消息加入记忆
        self.memory.add({"role": "user", "content": user_input})

        # 2. ReAct循环
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")

            # 回调
            if "on_iteration_start" in self.callback:
                self.callback["on_iteration_start"](iteration)

            # 2.1 获取记忆中的消息
            messages = self.memory.get_messages()
            response = await self.llm.achat(messages, tools=self.registry.get_tools())

            # 2.2 处理 LLM 响应
            if not response["tool_calls"]:
                # 2.2.1 如果没有工具调用，说明任务完成
                self.memory.add({"role": "assistant", "content": response["content"]})
                if self.verbose:
                    print(f"最终回答: {response['content']}")
                return response["content"]

            # 2.3 有 tool_calls，先把 assistant 消息加入记忆
            # 注意：这条消息带 tool_calls 字段，API 需要它来对应 tool 结果
            self.memory.add({"role": "assistant", "content": response["content"], "tool_calls": response["tool_calls"]})
            if self.verbose:
                for tc in response["tool_calls"]:
                    print(f"调用工具: {tc['function']['name']}({tc['function']['arguments'][:60]})")

            # 2.4 执行工具调用 （LLM 可能一次调多个工具）
            tool_tasks = [self._execute_tool(tc) for tc in response["tool_calls"]]
            tool_results = await asyncio.gather(*tool_tasks)

            # 2.5 将工具结果加入记忆
            for tool_message in tool_results:
                self.memory.add(tool_message)
                if self.verbose:
                    print(f"工具结果: {tool_message['content'][:80]}")

            # 2.6 进入下一轮
        # 3. 超过最大迭代次数，抛出异常
        raise MaxIterationsError(f"超过最大迭代次数 {self.max_iterations}，任务未完成。")
    async def _execute_tool(self, tool_call: dict) -> dict:
        # 解析工具调用
        tool_name = tool_call["function"]["name"]
        arguments = tool_call["arguments"]
        tool_call_id = tool_call["id"]

        # 回调
        if "on_tool_call_start" in self.callback:
            self.callback["on_tool_call_start"](tool_name, arguments)

        # 找工具
        tool_obj = self.registry.get_tool(tool_name)

        if tool_obj is None:
            # 工具不存在，返回错误消息
            result = f"工具 '{tool_name}' 不存在，可用工具：{self.registry.names()}"
            self._record_failure(tool_name)
        else:
            result = await tool_obj.ainvoke(arguments)

            if any(keyword in result.lower() for keyword in ["error", "exception", "failed"]):
                self._record_failure(tool_name)
            else:
                self._reset_count[tool_name] = 0  # 成功调用，重置计数器

        # 回调
        if "on_tool_call_end" in self.callback:
            self.callback["on_tool_call_end"](tool_name, arguments, result)

        # 返回工具结果消息
        return {
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call_id,
        }
    def _record_failure(self, tool_name: str):
        # 记录工具调用失败
        if tool_name not in self._failure_count:
            self._failure_count[tool_name] = 0
        self._failure_count[tool_name] += 1

        if self._failure_count[tool_name] >= 3:
            raise ConsecutiveFailureError(f"工具 '{tool_name}' 连续调用失败 {self._failure_count[tool_name]} 次。")
    def run(self, user_input: str) -> str:
        # 同步调用异步方法
        return asyncio.run(self.arun(user_input))
    
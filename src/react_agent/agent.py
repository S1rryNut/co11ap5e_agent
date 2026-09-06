import asyncio
from typing import Callable
from react_agent.llm import LLMClient
from react_agent.tool import Tool, ToolRegistry
from react_agent.memory.base import BaseMemory
from react_agent.memory.short_term import ShortTermMemory
from react_agent.utils.json_parser import parse_json_robust
from react_agent.errors import (
    MaxIterationsError,
    ConsecutiveFailureError,
)

# 摘要用的 system prompt
SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要器。请把用户提供的对话压缩成结构化摘要。

严格按以下 JSON 格式输出，不要输出其他内容：
{
  "summary": "对话过程的简要摘要（200字以内）",
  "facts": ["已确认的事实1", "已确认的事实2"],
  "preferences": ["用户明确说过的偏好1", "用户明确说过的偏好2"],
  "decisions": ["已经做出的决定1", "已经做出的决定2"],
  "open_questions": ["尚未解决的问题1", "尚未解决的问题2"]
}

规则：
1. 只保留原文明确出现的信息，不要推测、不要补充
2. 没有的字段填空数组 []
3. facts 是客观事实（如"用户在用 DeepSeek API"），不是观点
4. preferences 是用户明确表达的喜好（如"用户要求代码自己写"）
5. decisions 是已经做出的选择（如"选择了 Qdrant 作为向量库"）
6. open_questions 是还没解决的问题
7. summary 写对话过程，不要重复 facts 里的内容"""


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        memory: BaseMemory | None = None,
        max_iterations: int = 10,
        system_prompt: str | None = None,
        verbose: bool = False,
        callback: Callable | None = None,
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
        self._failure_count: dict[str, int] = {}  # 连续失败计数器

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

            # 2.0 检查是否需要压缩记忆
            if isinstance(self.memory, ShortTermMemory) and self.memory.should_compress():
                await self._compress_memory()

            # 2.1 获取记忆中的消息
            messages = self.memory.get_messages()
            response = await self.llm.achat(messages, tools=self.registry.get_schemas())

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
        arguments = tool_call["function"]["arguments"]
        tool_call_id = tool_call["id"]

        # 回调
        if "on_tool_call_start" in self.callback:
            self.callback["on_tool_call_start"](tool_name, arguments)

        # 找工具
        tool_obj = self.registry.get(tool_name)

        if tool_obj is None:
            # 工具不存在，返回错误消息
            result = f"工具 '{tool_name}' 不存在，可用工具：{self.registry.names()}"
            self._record_failure(tool_name)
        else:
            result = await tool_obj.ainvoke(arguments)

            if result.startswith("[TOOL_ERROR]"):
                self._record_failure(tool_name)
            else:
                self._failure_count[tool_name] = 0  # 成功调用，重置计数器

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

    async def _compress_memory(self):
        """压缩记忆：把旧消息摘要成结构化记忆。

        流程：
        1. 获取需要被压缩的旧消息
        2. 调用 LLM 生成结构化摘要
        3. 解析 JSON，应用压缩
        """
        old_messages = self.memory.get_messages_to_compress()
        if not old_messages:
            return

        if self.verbose:
            print(f"[记忆压缩] 压缩 {len(old_messages)} 条旧消息...")

        # 构建摘要请求（不带 tools，纯文本任务）
        summary_messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            *old_messages,
            {"role": "user", "content": "请总结以上对话，按要求的 JSON 格式输出。"},
        ]

        try:
            response = await self.llm.achat(summary_messages, temperature=0.3)
            parsed = parse_json_robust(response["content"])

            if parsed and isinstance(parsed, dict):
                self.memory.apply_compression(
                    old_messages=old_messages,
                    summary=parsed.get("summary", ""),
                    structured_updates={
                        "facts": parsed.get("facts", []),
                        "preferences": parsed.get("preferences", []),
                        "decisions": parsed.get("decisions", []),
                        "open_questions": parsed.get("open_questions", []),
                    },
                )
                if self.verbose:
                    stats = self.memory.get_stats()
                    print(f"[记忆压缩] 完成，当前 token: {stats['current_tokens']}/{stats['max_tokens']}")
            else:
                # JSON 解析失败，退化为纯文本摘要
                self.memory.apply_compression(
                    old_messages=old_messages,
                    summary=response["content"][:500],
                )
                if self.verbose:
                    print("[记忆压缩] JSON 解析失败，已保存为纯文本摘要")
        except Exception as e:
            # 压缩失败不影响主流程，打印日志后继续
            if self.verbose:
                print(f"[记忆压缩] 失败: {e}")

    def run(self, user_input: str) -> str:
        # 同步调用异步方法
        return asyncio.run(self.arun(user_input))

import asyncio
from typing import Callable
from react_agent.llm import LLMClient
from react_agent.tool import Tool, ToolRegistry
from react_agent.memory.base import BaseMemory
from react_agent.memory.short_term import ShortTermMemory
from react_agent.memory.working import WorkingMemory
from react_agent.memory.long_term import LongTermMemory
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
        working_memory: WorkingMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        max_iterations: int = 10,
        system_prompt: str | None = None,
        verbose: bool = False,
        callback: Callable | None = None,
        auto_save_long_term: bool = True,
    ):
        self.llm = llm
        self.registry = ToolRegistry()
        for tool in tools:
            self.registry.register(tool)
        self.memory = memory or ShortTermMemory(system_prompt=system_prompt)
        self.working_memory = working_memory
        self.long_term_memory = long_term_memory
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.callback = callback or {}
        self.auto_save_long_term = auto_save_long_term
        self._failure_count: dict[str, int] = {}

    async def arun(self, user_input: str) -> str:
        # 1. 设置工作记忆目标
        if self.working_memory is not None:
            self.working_memory.set_goal(user_input)

        # 2. 用户消息加入短期记忆
        self.memory.add({"role": "user", "content": user_input})

        # 3. ReAct循环
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")

            if "on_iteration_start" in self.callback:
                self.callback["on_iteration_start"](iteration)

            # 3.0 检查是否需要压缩记忆
            if isinstance(self.memory, ShortTermMemory) and self.memory.should_compress():
                await self._compress_memory()

            # 3.1 获取记忆中的消息，注入长期/工作记忆上下文
            messages = self._build_messages(user_input)
            response = await self.llm.achat(messages, tools=self.registry.get_schemas())

            # 3.2 处理 LLM 响应
            if not response["tool_calls"]:
                # 任务完成
                self.memory.add({"role": "assistant", "content": response["content"]})
                if self.verbose:
                    print(f"最终回答: {response['content']}")

                # 写入长期记忆
                await self._save_to_long_term(user_input, response["content"])

                # 标记工作记忆完成
                if self.working_memory is not None:
                    self.working_memory.mark_complete()

                return response["content"]

            # 3.3 有 tool_calls，加入记忆
            self.memory.add({"role": "assistant", "content": response["content"], "tool_calls": response["tool_calls"]})
            if self.verbose:
                for tc in response["tool_calls"]:
                    print(f"调用工具: {tc['function']['name']}({tc['function']['arguments'][:60]})")

            # 3.4 执行工具调用
            tool_tasks = [self._execute_tool(tc) for tc in response["tool_calls"]]
            tool_results = await asyncio.gather(*tool_tasks)

            # 3.5 工具结果加入记忆和工作记忆
            for tool_message in tool_results:
                self.memory.add(tool_message)
                if self.verbose:
                    print(f"工具结果: {tool_message['content'][:80]}")
                # 记录到工作记忆
                if self.working_memory is not None:
                    self.working_memory.add_result(
                        f"{tool_message.get('tool_call_id', '')}: {tool_message['content'][:100]}"
                    )

        # 4. 超过最大迭代次数
        raise MaxIterationsError(f"超过最大迭代次数 {self.max_iterations}，任务未完成。")

    def _build_messages(self, current_query: str) -> list[dict]:
        """构建最终发送给 LLM 的消息列表。

        在短期记忆的消息基础上，把长期记忆和工作记忆上下文注入 system prompt。
        """
        messages = self.memory.get_messages()

        # 构建额外上下文（长期记忆 + 工作记忆）
        extra_context_parts = []

        if self.long_term_memory is not None:
            lt_ctx = self.long_term_memory.build_context(query=current_query)
            if lt_ctx:
                extra_context_parts.append(lt_ctx)

        if self.working_memory is not None:
            wm_ctx = self.working_memory.build_context()
            if wm_ctx:
                extra_context_parts.append(wm_ctx)

        if not extra_context_parts:
            return messages

        extra_context = "\n\n".join(extra_context_parts)

        # 找到 system 消息，注入额外上下文
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                messages[i] = {
                    "role": "system",
                    "content": msg["content"] + "\n\n" + extra_context,
                }
                return messages

        # 没有 system 消息，在最前面加一个
        return [{"role": "system", "content": extra_context}] + messages

    async def _save_to_long_term(self, user_input: str, answer: str):
        """任务完成后，把对话摘要写入长期记忆。"""
        if self.long_term_memory is None:
            return

        try:
            # 用 LLM 生成对话摘要
            summary_messages = [
                {"role": "system", "content": "请用一句话总结这段对话的核心内容和关键点。"},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer[:500]},
            ]
            response = await self.llm.achat(summary_messages, temperature=0.3, max_tokens=200)
            summary = response["content"].strip()

            self.long_term_memory.add_conversation_summary(
                summary=summary,
                key_points=[user_input[:100]],
            )

            if self.auto_save_long_term:
                self.long_term_memory.save()

            if self.verbose:
                print(f"[长期记忆] 已保存对话摘要: {summary[:60]}...")
        except Exception as e:
            if self.verbose:
                print(f"[长期记忆] 保存失败: {e}")

    async def _execute_tool(self, tool_call: dict) -> dict:
        tool_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        tool_call_id = tool_call["id"]

        if "on_tool_call_start" in self.callback:
            self.callback["on_tool_call_start"](tool_name, arguments)

        tool_obj = self.registry.get(tool_name)

        if tool_obj is None:
            result = f"工具 '{tool_name}' 不存在，可用工具：{self.registry.names()}"
            self._record_failure(tool_name)
        else:
            result = await tool_obj.ainvoke(arguments)

            if result.startswith("[TOOL_ERROR]"):
                self._record_failure(tool_name)
            else:
                self._failure_count[tool_name] = 0

        if "on_tool_call_end" in self.callback:
            self.callback["on_tool_call_end"](tool_name, arguments, result)

        return {
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call_id,
        }

    def _record_failure(self, tool_name: str):
        if tool_name not in self._failure_count:
            self._failure_count[tool_name] = 0
        self._failure_count[tool_name] += 1

        if self._failure_count[tool_name] >= 3:
            raise ConsecutiveFailureError(f"工具 '{tool_name}' 连续调用失败 {self._failure_count[tool_name]} 次。")

    async def _compress_memory(self):
        """压缩记忆：把旧消息摘要成结构化记忆，并写入长期记忆。"""
        old_messages = self.memory.get_messages_to_compress()
        if not old_messages:
            return

        if self.verbose:
            print(f"[记忆压缩] 压缩 {len(old_messages)} 条旧消息...")

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

                # 写入长期记忆
                if self.long_term_memory is not None:
                    for fact in parsed.get("facts", []):
                        self.long_term_memory.add_fact(fact)
                    for pref in parsed.get("preferences", []):
                        self.long_term_memory.add_preference(pref)
                    for dec in parsed.get("decisions", []):
                        self.long_term_memory.add_decision(dec)
                    if parsed.get("summary"):
                        self.long_term_memory.add_conversation_summary(
                            summary=parsed["summary"],
                            key_points=parsed.get("facts", [])[:3],
                        )
                    if self.auto_save_long_term:
                        self.long_term_memory.save()

                if self.verbose:
                    stats = self.memory.get_stats()
                    print(f"[记忆压缩] 完成，当前 token: {stats['current_tokens']}/{stats['max_tokens']}")
            else:
                self.memory.apply_compression(
                    old_messages=old_messages,
                    summary=response["content"][:500],
                )
                if self.verbose:
                    print("[记忆压缩] JSON 解析失败，已保存为纯文本摘要")
        except Exception as e:
            if self.verbose:
                print(f"[记忆压缩] 失败: {e}")

    def run(self, user_input: str) -> str:
        return asyncio.run(self.arun(user_input))

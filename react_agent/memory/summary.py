from react_agent.memory.base import BaseMemory

# 摘要用的 prompt，约束 LLM 怎么压缩
SUMMARY_PROMPT = """你是对话摘要器。把下面的对话压缩成一段摘要。

必须保留：
1. 用户的个人信息和偏好
2. 用户做出的决定
3. 未解决的问题

不要添加原文没有的信息。
100 字以内。

对话内容：
{conversation}
"""

class SummaryMemory(BaseMemory):
    def __init__(
        self,
        max_tokens:int = 4000,
        keep_recent: int = 6,
        llm_client = None,
        system_prompt: str = None,
    ):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.llm = llm_client # LLMClient 实例，用来生成摘要

        self._messages = list[dict] = []
        self._summary: str | None = None  # 压缩后的摘要

        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def get_messages(self) -> list[dict]:
        from react_agent.utils.token_counter import count_messages_tokens

        # 1.把 system 消息和其他消息分开
        system_msg = [m for m in self._messages if m["role"] == "system"]
        other_msgs = [m for m in self._messages if m["role"] != "system"]

        total_tokens = count_messages_tokens(system_msg + other_msgs)

        # 2. 如果总 tokens 没超过 max_tokens * 1.5，就直接返回，不需要压缩
        if total_tokens <= self.max_tokens * 1.5:
            return system_msg + other_msgs

        # 3. 如果超过了，就先保留最近的 keep_recent 条消息，再压缩剩下的旧消息
        if len(other_msgs) > self.keep_recent:
            # 保留最近的 keep_recent 条消息
            recent_msgs = other_msgs[-self.keep_recent:]
            # 压缩剩下的旧消息
            old_msgs = other_msgs[:-self.keep_recent]
        else:
            recent_msgs = other_msgs
            old_msgs = []

        # 4. 如果没有旧消息，或者没有 LLM，就直接返回 system + recent
        if not old_msgs or self.llm is None:
            return system_msg + recent_msgs
        
        # 5. 把旧消息转成文本
        conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_msgs])

        # 6. 用 LLM 生成摘要
        prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
        response = self.llm.chat([{"role": "user", "content": prompt}])
        self._summary = response["content"]

        # 7. 创建摘要消息
        summary_msg = {"role": "system", "content": f"以下是之前对话的摘要: {self._summary}"}

        # 8. 返回 system + summary + recent
        self._messages = system_msg + [summary_msg] + recent_msgs

        return self._messages

    def clear(self):
        # 只保留 system 消息，清空其他消息和摘要
        system_msg = [m for m in self._messages if m["role"] == "system"]
        self._messages = system_msg
        self._summary = None
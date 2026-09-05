"""
摘要压缩记忆。

总 token 超过 max_tokens * 1.5 时触发压缩：
- 保留最近 keep_recent 条原始消息
- 更早的消息发给 LLM 生成摘要
- 摘要作为 system 消息放在最前面
"""
from react_agent.memory.base import BaseMemory


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
    """摘要压缩记忆。"""

    def __init__(
        self,
        max_tokens: int = 4000,
        keep_recent: int = 6,
        llm_client=None,
        system_prompt: str = "",
    ):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.llm = llm_client
        self._messages: list[dict] = []
        self._summary: str | None = None

        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def add(self, message: dict):
        self._messages.append(message)

    def get_messages(self) -> list[dict]:
        from react_agent.utils.token_counter import count_messages_tokens

        system_msgs = [m for m in self._messages if m["role"] == "system"]
        other_msgs = [m for m in self._messages if m["role"] != "system"]

        total = count_messages_tokens(system_msgs + other_msgs)

        # 没超过阈值（1.5 倍缓冲），直接返回
        if total <= self.max_tokens * 1.5:
            return system_msgs + other_msgs

        # 保留最近 keep_recent 条
        if len(other_msgs) > self.keep_recent:
            recent = other_msgs[-self.keep_recent:]
            old = other_msgs[:-self.keep_recent]
        else:
            recent = other_msgs
            old = []

        # 没有旧消息或没有 LLM，退化成滑动窗口
        if not old or self.llm is None:
            return system_msgs + recent

        # 旧消息转成文本
        conversation_text = "\n".join(
            f"{m['role']}: {m.get('content', '')}" for m in old
        )

        # 调 LLM 生成摘要
        prompt = SUMMARY_PROMPT.format(conversation=conversation_text)
        resp = self.llm.chat([{"role": "user", "content": prompt}])
        self._summary = resp["content"]

        # 摘要作为 system 消息
        summary_msg = {"role": "system", "content": f"以下是之前对话的摘要：\n{self._summary}"}

        # 替换旧消息
        self._messages = system_msgs + [summary_msg] + recent

        return self._messages

    def clear(self):
        system = [m for m in self._messages if m["role"] == "system"]
        self._messages = system
        self._summary = None

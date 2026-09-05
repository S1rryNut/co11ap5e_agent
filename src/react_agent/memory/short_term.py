"""
滑动窗口记忆。

保留最近的消息，总 token 不超过 max_tokens。
- system 消息永远保留
- 从后往前累加，超过 max_tokens 就截断
"""
from react_agent.memory.base import BaseMemory
from react_agent.utils.token_counter import count_messages_tokens


class ShortTermMemory(BaseMemory):
    """滑动窗口记忆。"""

    def __init__(self, max_tokens: int = 2000, system_prompt: str = ""):
        self.max_tokens = max_tokens
        self._messages: list[dict] = []

        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def add(self, message: dict):
        self._messages.append(message)

    def get_messages(self) -> list[dict]:
        # system 消息永远保留
        system_msgs = [m for m in self._messages if m["role"] == "system"]
        other_msgs = [m for m in self._messages if m["role"] != "system"]

        # 从后往前累加 token
        result = []
        total_tokens = count_messages_tokens(system_msgs)

        for msg in reversed(other_msgs):
            msg_tokens = count_messages_tokens([msg])
            if total_tokens + msg_tokens > self.max_tokens:
                break
            result.insert(0, msg)
            total_tokens += msg_tokens

        return system_msgs + result

    def clear(self):
        system = [m for m in self._messages if m["role"] == "system"]
        self._messages = system

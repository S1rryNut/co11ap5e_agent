from react_agent.memory.base import BaseMemory
from react_agent.utils.token_counter import count_messages_tokens

class ShortTermMemory(BaseMemory):

    def __init__(self, max_tokens: int = 2000, system_prompt: str = None):
        self.max_tokens = max_tokens
        self._messages = list[dict] = []

        # 如果传了 system_prompt，初始化时就加进去
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def add(self, message: dict):
        # 添加一条消息到记忆中
        self._messages.append(message)

    def get_messages(self) -> list[dict]:
        # 1.把 system 消息和其他消息分开
        system_msg = [m for m in self._messages if m["role"] == "system"]
        other_msgs = [m for m in self._messages if m["role"] != "system"]

        # 2.从后往前遍历其他消息，直到超过 max_tokens 为止
        result = []
        total_tokens = count_messages_tokens(system_msg)

        for msg in reversed(other_msgs):
            msg_tokens = count_messages_tokens([msg])
            # 如果加上这条消息后超过了 max_tokens，就停止添加
            if total_tokens + msg_tokens > self.max_tokens:
                break
            # 如果没超过，就加上这条消息
            result.insert(0, msg)
            total_tokens += msg_tokens

        # 3.返回结果，包含 system 消息和其他消息
        return system_msg + result

    def clear(self):
        # 只保留 system 消息，清空其他消息
        system_msg = [m for m in self._messages if m["role"] == "system"]
        self._messages = system_msg
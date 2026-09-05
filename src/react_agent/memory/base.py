"""
记忆抽象基类。

所有记忆实现都要实现 add / get_messages / clear 三个方法。
messages 格式统一为 OpenAI 格式：
    {"role": "system"|"user"|"assistant"|"tool", "content": str, ...}
"""
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """记忆抽象基类。"""

    @abstractmethod
    def add(self, message: dict):
        """添加一条消息。"""

    @abstractmethod
    def get_messages(self) -> list[dict]:
        """返回要发给 LLM 的消息列表（可能经过截断/摘要）。"""

    @abstractmethod
    def clear(self):
        """清空记忆。"""

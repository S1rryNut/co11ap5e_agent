from abc import ABC, abstractmethod

class BaseMemory(ABC):
    """记忆抽象基类。

    所有记忆实现都要实现这三个方法。
    messages 格式统一为 OpenAI 格式：
        {"role": "system"|"user"|"assistant"|"tool", "content": str, ...}
    """

    @abstractmethod
    def add(self, message: dict):
        """添加一条消息到记忆中。"""

    @abstractmethod
    def get_messages(self) -> list[dict]:
        """获取记忆中的所有消息。"""

    @abstractmethod
    def clear(self):
        """清空记忆。"""
        

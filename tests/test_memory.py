"""
记忆系统单元测试
"""
import pytest
from react_agent.memory import ShortTermMemory, BaseMemory


class TestShortTermMemory:
    """测试滑动窗口记忆"""

    def test_add_and_get(self):
        mem = ShortTermMemory(max_tokens=1000)
        mem.add({"role": "user", "content": "你好"})
        mem.add({"role": "assistant", "content": "你好！"})

        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "你好"
        assert messages[1]["content"] == "你好！"

    def test_system_prompt_always_kept(self):
        """system 消息永远保留"""
        mem = ShortTermMemory(max_tokens=10, system_prompt="你是助手")
        for i in range(20):
            mem.add({"role": "user", "content": f"消息 {i} " * 10})

        messages = mem.get_messages()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "你是助手"

    def test_truncation(self):
        """超过 max_tokens 时截断旧消息"""
        mem = ShortTermMemory(max_tokens=50)
        for i in range(10):
            mem.add({"role": "user", "content": f"这是第 {i} 条很长的消息内容" * 3})

        messages = mem.get_messages()
        # 应该只保留最近的几条
        assert len(messages) < 10
        # 最后一条应该是最新的
        assert "第 9 条" in messages[-1]["content"]

    def test_clear(self):
        mem = ShortTermMemory(system_prompt="你是助手")
        mem.add({"role": "user", "content": "你好"})
        mem.clear()

        messages = mem.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_is_base_memory(self):
        mem = ShortTermMemory()
        assert isinstance(mem, BaseMemory)

"""
Token 计数单元测试
"""
import pytest
from react_agent.utils.token_counter import (
    count_tokens,
    estimate_tokens,
    count_messages_tokens,
)


class TestCountTokens:
    """测试 Token 计数"""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_chinese(self):
        # 中文 4 个字，估算约 6 token
        result = count_tokens("你好世界")
        assert result > 0

    def test_english(self):
        result = count_tokens("hello world")
        assert result > 0

    def test_mixed(self):
        result = count_tokens("你好 world 123")
        assert result > 0

    def test_long_text(self):
        text = "这是一段很长的文本" * 100
        result = count_tokens(text)
        assert result > 100


class TestEstimateTokens:
    """测试估算函数"""

    def test_chinese_ratio(self):
        # 中文 10 字 ≈ 15 token
        result = estimate_tokens("你好世界你好世界你好")
        assert 10 <= result <= 20

    def test_english_ratio(self):
        # 英文单词数 * 1.3 + 字母外字符
        result = estimate_tokens("hello world foo bar")
        assert result > 0

    def test_min_one(self):
        assert estimate_tokens("a") >= 1


class TestCountMessagesTokens:
    """测试消息列表计数"""

    def test_empty_list(self):
        assert count_messages_tokens([]) == 0

    def test_basic_messages(self):
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        result = count_messages_tokens(messages)
        assert result > 0

    def test_tool_calls_counted(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"arguments": '{"a": 1, "b": 2}'}}],
            }
        ]
        result = count_messages_tokens(messages)
        assert result > 0

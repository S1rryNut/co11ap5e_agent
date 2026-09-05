"""
容错 JSON 解析器单元测试
"""
import pytest
from react_agent.utils.json_parser import parse_json_robust, extract_json_block


class TestParseJsonRobust:
    """测试容错 JSON 解析"""

    def test_standard_json(self):
        result = parse_json_robust('{"a": 1, "b": 2}')
        assert result == {"a": 1, "b": 2}

    def test_json_array(self):
        result = parse_json_robust('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_with_code_block(self):
        text = '```json\n{"a": 1}\n```'
        result = parse_json_robust(text)
        assert result == {"a": 1}

    def test_with_code_block_no_lang(self):
        text = '```\n{"a": 1}\n```'
        result = parse_json_robust(text)
        assert result == {"a": 1}

    def test_with_trailing_text(self):
        text = '好的，结果是：{"a": 1}，希望对你有帮助'
        result = parse_json_robust(text)
        assert result == {"a": 1}

    def test_trailing_comma(self):
        result = parse_json_robust('{"a": 1, "b": 2,}')
        assert result == {"a": 1, "b": 2}

    def test_single_quotes(self):
        result = parse_json_robust("{'a': 1, 'b': 2}")
        assert result == {"a": 1, "b": 2}

    def test_empty_string(self):
        assert parse_json_robust("") is None
        assert parse_json_robust("   ") is None

    def test_not_json(self):
        assert parse_json_robust("你好世界") is None

    def test_nested_json(self):
        text = '{"a": {"b": [1, 2, 3]}}'
        result = parse_json_robust(text)
        assert result == {"a": {"b": [1, 2, 3]}}


class TestExtractJsonBlock:
    """测试 JSON 提取"""

    def test_extract_from_code_block(self):
        text = '```json\n{"a": 1}\n```'
        assert extract_json_block(text) == '{"a": 1}'

    def test_extract_from_text(self):
        text = '结果：{"a": 1}，完毕'
        assert extract_json_block(text) == '{"a": 1}'

    def test_extract_array(self):
        text = '列表：[1, 2, 3]，结束'
        assert extract_json_block(text) == '[1, 2, 3]'

    def test_no_json(self):
        assert extract_json_block("你好") is None

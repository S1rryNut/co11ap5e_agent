"""
工具系统单元测试
"""
import pytest
from react_agent.tool import Tool, ToolRegistry, tool


class TestToolSchema:
    """测试工具 Schema 自动生成"""

    def test_basic_types(self):
        """测试基本类型的 schema 生成"""
        @tool
        def add(a: int, b: int) -> int:
            """两数相加。"""
            return a + b

        schema = add.schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert "a" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
        assert "a" in schema["function"]["parameters"]["required"]
        assert "b" in schema["function"]["parameters"]["required"]

    def test_optional_param(self):
        """测试有默认值的参数不在 required 里"""
        @tool
        def search(query: str, max_results: int = 3) -> str:
            """搜索。"""
            return query

        schema = search.schema
        props = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]
        assert props["query"]["type"] == "string"
        assert props["max_results"]["type"] == "integer"
        assert "query" in required
        assert "max_results" not in required

    def test_list_param(self):
        """测试 list 参数"""
        from typing import List

        @tool
        def sort_items(items: List[int]) -> list:
            """排序。"""
            return sorted(items)

        schema = sort_items.schema
        props = schema["function"]["parameters"]["properties"]
        assert props["items"]["type"] == "array"
        assert props["items"]["items"]["type"] == "integer"

    def test_string_type_default(self):
        """没有类型注解的参数默认是 string"""
        @tool
        def echo(x) -> str:
            """回显。"""
            return str(x)

        schema = echo.schema
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "string"

    def test_description_from_docstring(self):
        """description 来自 docstring"""
        @tool
        def add(a: int, b: int) -> int:
            """两数相加，返回和。"""
            return a + b

        assert add.schema["function"]["description"] == "两数相加，返回和。"


class TestToolInvoke:
    """测试工具执行"""

    def test_invoke_with_json_string(self):
        @tool
        def add(a: int, b: int) -> int:
            """两数相加。"""
            return a + b

        result = add.invoke('{"a": 3, "b": 5}')
        assert result == "8"

    def test_invoke_with_dict(self):
        @tool
        def add(a: int, b: int) -> int:
            """两数相加。"""
            return a + b

        result = add.invoke({"a": 3, "b": 5})
        assert result == "8"

    def test_invoke_with_bad_json(self):
        """测试坏 JSON 返回错误信息，不抛异常"""
        @tool
        def add(a: int, b: int) -> int:
            """两数相加。"""
            return a + b

        result = add.invoke("不是 json")
        assert "解析失败" in result or "出错" in result

    def test_invoke_tool_error(self):
        """工具执行出错返回错误信息，不抛异常"""
        @tool
        def divide(a: int, b: int) -> float:
            """除法。"""
            return a / b

        result = divide.invoke('{"a": 1, "b": 0}')
        assert "出错" in result

    def test_invoke_non_string_result(self):
        """非字符串结果自动转 JSON 字符串"""
        @tool
        def get_dict() -> dict:
            """返回字典。"""
            return {"key": "value"}

        result = get_dict.invoke("{}")
        assert '"key"' in result
        assert '"value"' in result


class TestToolRegistry:
    """测试工具注册表"""

    def test_register_and_get(self):
        registry = ToolRegistry()

        @tool
        def add(a: int, b: int) -> int:
            """相加"""
            return a + b

        registry.register(add)
        assert registry.get("add") is not None
        assert len(registry.get_schemas()) == 1

    def test_register_plain_function(self):
        """可以直接注册普通函数（自动包成 Tool）"""
        registry = ToolRegistry()

        def multiply(a: int, b: int) -> int:
            """相乘。"""
            return a * b

        registry.register(multiply)
        assert registry.get("multiply") is not None
        assert registry.names() == ["multiply"]

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        assert registry.get("not_exist") is None

    def test_multiple_tools(self):
        registry = ToolRegistry()

        @tool
        def add(a: int, b: int) -> int:
            """相加"""
            return a + b

        @tool
        def sub(a: int, b: int) -> int:
            """相减"""
            return a - b

        registry.register(add)
        registry.register(sub)
        assert len(registry) == 2
        assert set(registry.names()) == {"add", "sub"}

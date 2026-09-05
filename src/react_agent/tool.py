"""
工具系统

@tool 装饰器把普通函数变成 LLM 可调用的工具。
自动从类型注解和 docstring 生成 OpenAI Function Calling JSON Schema。
"""
import inspect
import json
import asyncio
from typing import Any, Callable, get_type_hints, get_origin, get_args, Literal, Union
from pydantic import BaseModel


# Python 基本类型 → JSON Schema 类型
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _type_to_schema(py_type) -> dict:
    """把任意 Python 类型转成 JSON Schema。

    支持：
    - 基本类型: str/int/float/bool/list/dict
    - list[X]: 转成 array + items
    - Optional[X] (= Union[X, None]): 转成 anyOf
    - Literal["a", "b"]: 转成 enum
    - Pydantic Model: 递归展开成 nested object
    """
    # 基本类型
    if py_type in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[py_type]}

    origin = get_origin(py_type)

    # list[X]
    if origin is list:
        args = get_args(py_type)
        items_schema = _type_to_schema(args[0]) if args else {}
        return {"type": "array", "items": items_schema}

    # Optional[X] = Union[X, None]
    if origin is Union:
        args = get_args(py_type)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return {"anyOf": [_type_to_schema(non_none[0]), {"type": "null"}]}

    # Literal["a", "b"]
    if origin is Literal:
        return {"type": "string", "enum": list(get_args(py_type))}

    # Pydantic Model
    if isinstance(py_type, type) and issubclass(py_type, BaseModel):
        properties = {}
        required = []
        for field_name, field in py_type.model_fields.items():
            properties[field_name] = _type_to_schema(field.annotation)
            if field.is_required():
                required.append(field_name)
        return {"type": "object", "properties": properties, "required": required}

    # 兜底
    return {"type": "string"}


class Tool:
    """包装一个函数，使其成为 LLM 可调用的工具。"""

    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.description = inspect.getdoc(func) or ""
        self._schema = self._build_schema()

    def _build_schema(self) -> dict:
        """从函数签名自动生成 JSON Schema。"""
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            annotation = hints.get(param_name, str)
            properties[param_name] = _type_to_schema(annotation)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @property
    def schema(self) -> dict:
        """返回 OpenAI Function Calling 格式的 schema。"""
        return self._schema

    async def ainvoke(self, arguments: str | dict) -> str:
        """异步执行工具。

        arguments 可以是 JSON 字符串或 dict。
        返回字符串结果（LLM 只能理解字符串）。
        """
        # 1. 解析参数
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return f"参数解析失败: {e}"

        # 2. 执行（同步函数丢线程池，避免阻塞事件循环）
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.func, **arguments),
                timeout=30,
            )
        except asyncio.TimeoutError:
            return "工具执行超时（30秒）"
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {e}"

        # 3. 结果转字符串
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        return result

    def invoke(self, arguments: str | dict) -> str:
        """同步执行工具。"""
        return asyncio.run(self.ainvoke(arguments))


def tool(func: Callable) -> Tool:
    """装饰器：把普通函数变成 Tool 对象。

    用法：
        @tool
        def add(a: int, b: int) -> int:
            '''两数相加。'''
            return a + b
    """
    return Tool(func)


class ToolRegistry:
    """工具注册表：管理一组工具，按名称查找。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool_obj: Tool | Callable) -> Tool:
        """注册一个工具。可以传 Tool 对象或普通函数。"""
        if not isinstance(tool_obj, Tool):
            tool_obj = Tool(tool_obj)
        self._tools[tool_obj.name] = tool_obj
        return tool_obj

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        """返回所有工具的 schema 列表，直接传给 OpenAI API。"""
        return [t.schema for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self):
        return len(self._tools)

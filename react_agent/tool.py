import inspect
import json
import asyncio
from typing import Any, Callable, get_type_hints, get_origin, get_args, Literal, Union
from pydantic import BaseModel
from react_agent.utils.json_parser import parse_json_robust


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
    # 将 Python 类型转换为 JSON Schema
    origin = get_origin(py_type)

    # 1：处理基本类型（str/int/float/bool/list/dict）
    if py_type in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[py_type]}
    
    # 2a: 处理 Literal 类型 Literal["a", "b"] → enum
    if origin is Literal:
        return {"type": "string", "enum": list(get_args(py_type))}

    # 2b: 处理 Union 类型 Optional[X] = Union[X, None] → anyOf
    if origin is Union:
        args = get_args(py_type)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return {
                "anyOf": [
                    _type_to_schema(non_none[0]),
                    {"type": "null"},
                ]
            }

    # 2c: 处理 List 类型 list[X] → {"type": "array", "items": {...}}
    if origin is list:
        args = get_args(py_type)
        items_schema = _type_to_schema(args[0]) if args else {}
        return {"type": "array", "items": items_schema}

    # 3：Pydantic Model → 递归展开
    if isinstance(py_type, type) and issubclass(py_type, BaseModel):
        properties = {}
        required = []
        for field_name, field in py_type.model_fields.items():
            properties[field_name] = _type_to_schema(field.annotation)
            if field.is_required():
                required.append(field_name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    # 兜底：不认识的类型当 string
    return {"type": "string"}

class Tool:
    # 工具类，封装了一个函数和它的参数类型信息 
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.description = inspect.getdoc(func) or None
        self._schema = self._build_schema()

    def _build_schema(self) -> dict:
        # 构建工具的 JSON Schema 从函数签名中提取参数类型信息
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # 跳过 self 参数 
            if param_name == "self":
                continue  

            # 获取参数类型，如果没有类型提示，默认使用 str
            param_type = hints.get(param_name, str)  # 默认类型为 str
            properties[param_name] = _type_to_schema(param_type)

            # 检查参数是否有默认值，如果没有，则认为是必填参数
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        return {
            "type": "object",
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
        # 返回工具的 JSON Schema
        return self._schema

    async def ainvoke(self, arguments: str | dict) -> str:
        # 异步调用工具函数 
        # 1.解析参数
        if isinstance(arguments, str):
            parsed = parse_json_robust(arguments)
            if parsed is None:
                return f"参数解析失败，无法解析 JSON: {arguments[:100]}"
            arguments = parsed


        # 2.调用函数
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.func, **arguments),
                timeout=30,
            )
        except asyncio.TimeoutError:
            return "工具调用超时"
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {e}"

        # 3. 结果转字符串（LLM 只理解字符串）
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False,default=str)
        return result

    def invoke(self, arguments: str | dict) -> str:
        # 同步调用异步方法
        return asyncio.run(self.ainvoke(arguments))

    def tool(func: Callable) -> "Tool":
        # 装饰器，将函数包装为 Tool 对象
        return Tool(func)

class ToolRegistry:
    # 工具注册表，管理所有工具
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool_obj: Tool | Callable) -> Tool:
        # 注册工具
        if not isinstance(tool_obj, Tool):
            tool_obj = Tool(tool_obj)
        self._tools[tool_obj.name] = tool_obj
        return tool_obj

    def get_tool(self, name: str) -> Tool | None:
        # 根据名称获取工具
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        # 获取所有工具的 JSON Schema
        return [t.schema for t in self._tools.values()]

    def names(self) -> list[str]:
        # 获取所有工具的名称
        return list(self._tools.keys())

    def __len__(self) -> int:
        # 工具数量
        return len(self._tools)
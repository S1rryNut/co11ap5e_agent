"""
容错 JSON 解析器。

LLM 输出的 arguments 经常不规范（带代码块、注释、尾随逗号、单引号），
标准 json.loads 会报错。这个模块用多种策略尝试解析。
"""
import json
import re
import ast


def extract_json_block(text: str) -> str | None:
    """从文本中提取 JSON 代码块或 JSON 片段。

    处理情况：
    1. ```json ... ``` 代码块
    2. ``` ... ``` 代码块（无语言标记）
    3. 第一个 { 到最后一个 } 之间的内容
    4. 第一个 [ 到最后一个 ] 之间的内容
    """
    if not text:
        return None

    # 情况 1: ```json ... ```
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 情况 2: ``` ... ```（无 json 标记）
    m = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 情况 3: 第一个 { 到最后一个 }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    # 情况 4: 第一个 [ 到最后一个 ]
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return None


def _fix_json(text: str) -> str:
    """尝试修复不规范的 JSON 字符串。"""
    fixed = text

    # 去掉 // 单行注释
    fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)

    # 去掉 /* ... */ 块注释
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)

    # 去掉尾随逗号（} 或 ] 前面的逗号）
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

    return fixed


def parse_json_robust(text: str) -> dict | list | None:
    """尝试用多种方式解析 JSON，全部失败返回 None。

    尝试顺序：
    1. 标准 json.loads
    2. 提取 JSON 代码块后再解析
    3. 去掉注释后解析
    4. 去掉尾随逗号后解析
    5. 单引号转双引号后解析
    6. ast.literal_eval 兜底
    """
    if not text or not text.strip():
        return None

    # 策略 1: 标准解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略 2: 先提取 JSON 块
    block = extract_json_block(text)
    if block and block != text:
        try:
            return json.loads(block)
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        block = text

    # 策略 3: 去掉注释和尾随逗号
    fixed = _fix_json(block)
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略 4: 单引号转双引号（最后手段，可能误伤）
    try:
        single_quoted = block.replace("'", '"')
        return json.loads(single_quoted)
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略 5: ast.literal_eval 兜底
    try:
        result = ast.literal_eval(block)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass

    return None

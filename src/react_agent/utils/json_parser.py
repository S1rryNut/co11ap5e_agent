import json
import re
import ast

# 从文本中提取 JSON 代码块或 JSON 片段。
def extract_json_block(text: str) -> str | None:
    if not text:
        return None

    # 1: ```json ... ```
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2: ``` ... ```（无 json 标记）
    m = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3: 第一个 { 到最后一个 }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    # 4: 第一个 [ 到最后一个 ]
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return None

# 尝试解析 JSON 字符串，如果失败则尝试修复后再解析。
def _fix_json(text: str) -> str:
    fixed = text

    # 1: 去掉 // 单行注释
    fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)

    # 2: 去掉 /* ... */ 块注释
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)

    # 3: 去掉尾随逗号（} 或 ] 前面的逗号）
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

    # 4: 单引号转双引号
    fixed = re.sub(r"'", '"', fixed)
    return fixed

def parse_json_robust(text: str) -> dict | list | None:
    # 尝试解析 JSON，如果失败则尝试修复后再解析。
    if not text or not text.strip():
        return None

    # 1: 标准解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2: 先提取 JSON 块
    block = extract_json_block(text)
    if block and block != text:
        try:
            return json.loads(block)
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        block = text

    # 3: 去掉注释
    fixed = _fix_json(block)
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # 4: 单引号转双引号（最后手段，可能误伤）
    try:
        single_quoted = block.replace("'", '"')
        return json.loads(single_quoted)
    except (json.JSONDecodeError, ValueError):
        pass

    # 5: ast.literal_eval 兜底
    try:
        result = ast.literal_eval(block)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass

    return None

"""
Token 计数工具。

优先用 tiktoken 精确计数，不支持的模型回退到估算。
"""
import re

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """优先用 tiktoken 精确计数，不支持的模型回退到估算。"""
    if not text:
        return 0

    if _HAS_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass

    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """估算 token 数（不依赖 tiktoken）。

    规则：
    - 中文字符：1 字 ≈ 1.5 token
    - 英文单词：1 词 ≈ 1.3 token
    - 数字/标点/其他：1 个 ≈ 1 token
    """
    if not text:
        return 0

    # 中文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))

    # 英文单词
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    english_letters = sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))

    # 其他字符
    other_chars = len(text) - chinese_chars - english_letters

    total = int(chinese_chars * 1.5 + english_words * 1.3 + other_chars * 1)
    return max(1, total)


def count_messages_tokens(messages: list[dict], model: str = "gpt-3.5-turbo") -> int:
    """计算一组消息的总 token 数。"""
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", ""), model)

        # tool_calls 里的 JSON 参数也要算（兼容嵌套 function.arguments 和拍平 arguments）
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                arguments = tc.get("arguments", "")
                if not arguments and "function" in tc:
                    arguments = tc["function"].get("arguments", "")
                total += count_tokens(arguments, model)

    return total

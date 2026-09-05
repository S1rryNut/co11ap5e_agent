import re

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    # 计算文本的 token 数量，使用 tiktoken 库
    if not text:
        return 0

    if not _HAS_TIKTOKEN:
        return None

    if _HAS_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            # 模型不认识（比如 deepseek-chat），回退到估算
            pass
    # 回退：估算
    return estimate_tokens(text)

def estimate_tokens(text: str) -> int:
    # 估算文本的 token 数量，简单按空格分词
    if not text:
        return 0
    # 计算中文字符和英文单词的数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'\b\w+\b', text))

    # 计算英文单词的长度总和
    english_letters = sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))
    other_chars = len(text) - chinese_chars - english_letters

    # 加权求和
    total = int(chinese_chars * 1.5 + english_words * 1.3 + other_chars * 1)
    return max(1, total)

def count_messages_tokens(messages: list[dict], model: str = "gpt-3.5-turbo") -> int:
    # 计算消息列表的总 token 数量
    total_tokens = 0
    for message in messages:
        # 计算每条消息的 content 字段的 token 数量
        total_tokens += count_tokens(message.get("content", ""), model)
        # 计算每条消息的 tool_calls 字段的 token 数量
        if "tool_calls" in message and message["tool_calls"]:
            for tool_call in message["tool_calls"]:
                total_tokens += count_tokens(tool_call.get("arguments", ""), model)
                
    return total_tokens
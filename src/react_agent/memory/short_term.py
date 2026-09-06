"""分层短期记忆。

三层结构：
1. 原始消息：最近 N 轮保留原文
2. 摘要记忆：更早的对话经 LLM 摘要后存储
3. 结构化记忆：facts / preferences / decisions / open_questions

动态压缩：token 超过阈值时触发，保留最近 N 轮原文，旧消息做摘要。
"""
from react_agent.memory.base import BaseMemory
from react_agent.utils.token_counter import count_messages_tokens


class ShortTermMemory(BaseMemory):

    def __init__(
        self,
        max_tokens: int = 2000,
        system_prompt: str = None,
        compress_threshold: float = 0.7,   # token 达到 max_tokens 的 70% 时触发压缩
        compress_target: float = 0.35,     # 压缩后目标占比 35%
        keep_recent_rounds: int = 3,       # 永远保留最近 3 轮原文
    ):
        self.max_tokens = max_tokens
        self.compress_threshold = compress_threshold
        self.compress_target = compress_target
        self.keep_recent_rounds = keep_recent_rounds

        self._messages: list[dict] = []
        self._summary: str = ""            # 历史摘要（渐进式追加）
        self._structured: dict = {         # 结构化记忆
            "facts": [],
            "preferences": [],
            "decisions": [],
            "open_questions": [],
        }

        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    # ============ 基础操作 ============

    def add(self, message: dict):
        self._messages.append(message)

    def clear(self):
        system_msg = [m for m in self._messages if m["role"] == "system"]
        self._messages = system_msg
        self._summary = ""
        self._structured = {"facts": [], "preferences": [], "decisions": [], "open_questions": []}

    # ============ 压缩判断 ============

    def should_compress(self) -> bool:
        """判断是否需要压缩。

        计算当前所有内容（system + 消息 + 摘要 + 结构化记忆）的 token 数，
        超过 max_tokens * compress_threshold 时返回 True。
        """
        system_msg = [m for m in self._messages if m["role"] == "system"]
        other_msgs = [m for m in self._messages if m["role"] != "system"]

        total = count_messages_tokens(system_msg + other_msgs)
        # 加上记忆上下文的 token
        memory_ctx = self._build_memory_context()
        if memory_ctx:
            total += count_messages_tokens([{"role": "system", "content": memory_ctx}])

        return total > self.max_tokens * self.compress_threshold

    def get_messages_to_compress(self) -> list[dict]:
        """获取需要被压缩的旧消息。

        保留最近 keep_recent_rounds 轮原文，更早的返回给调用方做摘要。
        如果总轮数 <= keep_recent_rounds，返回空列表（不需要压缩）。
        """
        other_msgs = [m for m in self._messages if m["role"] != "system"]
        rounds = self._group_into_rounds(other_msgs)

        if len(rounds) <= self.keep_recent_rounds:
            return []

        # 旧轮次（除了最近 N 轮）需要被压缩
        old_rounds = rounds[:-self.keep_recent_rounds]
        old_messages = []
        for r in old_rounds:
            old_messages.extend(r)
        return old_messages

    def apply_compression(
        self,
        old_messages: list[dict],
        summary: str,
        structured_updates: dict = None,
    ):
        """应用压缩结果。

        Args:
            old_messages: 被压缩的旧消息列表（用于删除）
            summary: LLM 生成的摘要文本
            structured_updates: 结构化记忆更新，格式如
                {"facts": [...], "preferences": [...], ...}
        """
        # 删除旧消息（用 id 精确匹配，避免误删）
        old_ids = {id(m) for m in old_messages}
        self._messages = [m for m in self._messages if id(m) not in old_ids]

        # 渐进式追加摘要（不是每次重新生成全部）
        if self._summary:
            self._summary = self._summary + "\n\n" + summary.strip()
        else:
            self._summary = summary.strip()

        # 合并结构化记忆（去重）
        if structured_updates:
            for key in ["facts", "preferences", "decisions", "open_questions"]:
                if key in structured_updates and structured_updates[key]:
                    existing = set(self._structured[key])
                    for item in structured_updates[key]:
                        item = item.strip()
                        if item and item not in existing:
                            self._structured[key].append(item)
                            existing.add(item)

    # ============ 获取消息 ============

    def get_messages(self) -> list[dict]:
        """获取最终发送给 LLM 的消息列表。

        结构：[enhanced_system] + [最近的原始消息（按轮次截断）]
        enhanced_system = 原始 system_prompt + 摘要 + 结构化记忆
        """
        system_msg = [m for m in self._messages if m["role"] == "system"]
        other_msgs = [m for m in self._messages if m["role"] != "system"]

        # 把记忆上下文注入 system prompt
        memory_context = self._build_memory_context()
        if memory_context and system_msg:
            enhanced = {
                "role": "system",
                "content": system_msg[0]["content"] + "\n\n" + memory_context,
            }
            system_msg = [enhanced]

        # 按轮次从后往前保留，直到超过 max_tokens
        rounds = self._group_into_rounds(other_msgs)
        result = []
        total_tokens = count_messages_tokens(system_msg)

        for round_msgs in reversed(rounds):
            round_tokens = count_messages_tokens(round_msgs)
            if total_tokens + round_tokens > self.max_tokens:
                break
            result = round_msgs + result
            total_tokens += round_tokens

        return system_msg + result

    # ============ 内部方法 ============

    def _build_memory_context(self) -> str:
        """构建记忆上下文文本，注入到 system prompt。"""
        parts = []

        if self._summary:
            parts.append(f"【历史对话摘要】\n{self._summary}")

        has_structured = any(self._structured[k] for k in self._structured)
        if has_structured:
            parts.append("【关键信息】")
            if self._structured["facts"]:
                parts.append(f"- 已确认事实：{'; '.join(self._structured['facts'])}")
            if self._structured["preferences"]:
                parts.append(f"- 用户偏好：{'; '.join(self._structured['preferences'])}")
            if self._structured["decisions"]:
                parts.append(f"- 已做决定：{'; '.join(self._structured['decisions'])}")
            if self._structured["open_questions"]:
                parts.append(f"- 未解决问题：{'; '.join(self._structured['open_questions'])}")

        return "\n".join(parts)

    def _group_into_rounds(self, messages: list[dict]) -> list[list[dict]]:
        """把消息按轮次分组。

        一轮 = 一条普通消息，或 一个 assistant(带 tool_calls) + 后续连续的 tool 消息。
        """
        rounds = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    j += 1
                rounds.append(messages[i:j])
                i = j
            else:
                rounds.append([msg])
                i += 1
        return rounds

    # ============ 调试用 ============

    def get_stats(self) -> dict:
        """获取记忆统计信息，用于调试。"""
        other_msgs = [m for m in self._messages if m["role"] != "system"]
        rounds = self._group_into_rounds(other_msgs)
        return {
            "total_messages": len(self._messages),
            "total_rounds": len(rounds),
            "summary_length": len(self._summary),
            "structured_counts": {k: len(v) for k, v in self._structured.items()},
            "current_tokens": count_messages_tokens(self.get_messages()),
            "max_tokens": self.max_tokens,
            "should_compress": self.should_compress(),
        }

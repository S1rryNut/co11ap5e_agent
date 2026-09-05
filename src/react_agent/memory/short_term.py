from react_agent.memory.base import BaseMemory
from react_agent.utils.token_counter import count_messages_tokens

class ShortTermMemory(BaseMemory):

    def __init__(self, max_tokens: int = 2000, system_prompt: str = None):
        self.max_tokens = max_tokens
        self._messages: list[dict] = []

        # 如果传了 system_prompt，初始化时就加进去
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def add(self, message: dict):
        # 添加一条消息到记忆中
        self._messages.append(message)

    def get_messages(self) -> list[dict]:
        # 获取当前记忆中的消息列表，按时间顺序（旧到新）
        system_msg = [m for m in self._messages if m["role"] == "system"]
        other_msgs = [m for m in self._messages if m["role"] != "system"]

        # 第一步：从后往前把消息分成"轮次"
        # 一轮 = 一条普通消息，或者 一个 assistant(带 tool_calls) + 后续连续的 tool 消息
        rounds = []
        i = len(other_msgs) - 1
        while i >= 0:
            msg = other_msgs[i]
            if msg["role"] == "tool":
                # 往前找到对应的 assistant（带 tool_calls）
                j = i
                while j >= 0 and other_msgs[j]["role"] == "tool":
                    j -= 1
                # j 现在是 assistant（带 tool_calls）的位置
                if j >= 0 and other_msgs[j]["role"] == "assistant" and other_msgs[j].get("tool_calls"):
                    # 一轮：assistant + 所有 tool 消息
                    round_msgs = other_msgs[j:i+1]
                    rounds.append(round_msgs)
                    i = j - 1
                else:
                    # 孤立的 tool 消息，跳过
                    i -= 1
            else:
                # 普通消息（user / assistant 不带 tool_calls）
                rounds.append([msg])
                i -= 1

        # 第二步：从后往前（rounds 已经是倒序的）按轮次保留，直到超过 max_tokens
        result = []
        total_tokens = count_messages_tokens(system_msg)

        for round_msgs in rounds:  # rounds 已经是从新到旧的顺序
            round_tokens = count_messages_tokens(round_msgs)
            if total_tokens + round_tokens > self.max_tokens:
                break  # 这一轮放不下，整个丢掉（不会出现半轮）
            result = round_msgs + result  # 插到前面
            total_tokens += round_tokens

        return system_msg + result



    def clear(self):
        # 只保留 system 消息，清空其他消息
        system_msg = [m for m in self._messages if m["role"] == "system"]
        self._messages = system_msg

"""长期记忆。

跨会话持久化，存储到 JSON 文件。
包含：
- user_profile: 用户画像（偏好、事实、习惯）
- conversation_summaries: 历史对话摘要列表
- decisions: 重要决定
- knowledge: 知识条目（从对话中提取的有用信息）

提供简单的关键词检索，后续可扩展为向量检索。
"""
import json
import os
from datetime import datetime


class LongTermMemory:

    def __init__(self, storage_path: str = "long_term_memory.json"):
        self.storage_path = storage_path
        self.data = {
            "user_profile": {
                "preferences": [],   # 用户偏好
                "facts": [],         # 关于用户的事实
                "habits": [],        # 用户习惯
            },
            "conversation_summaries": [],  # 历史对话摘要 [{date, summary, key_points}]
            "decisions": [],               # 重要决定 [{date, decision, context}]
            "knowledge": [],               # 知识条目 [{date, topic, content}]
        }
        self._load()

    # ============ 持久化 ============

    def _load(self):
        """从 JSON 文件加载记忆。"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # 合并，确保新字段存在
                for key in self.data:
                    if key in loaded:
                        self.data[key] = loaded[key]
            except (json.JSONDecodeError, IOError):
                pass  # 文件损坏时用空记忆

    def save(self):
        """保存记忆到 JSON 文件。"""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ============ 用户画像 ============

    def add_preference(self, preference: str):
        """添加用户偏好（去重）。"""
        if preference not in self.data["user_profile"]["preferences"]:
            self.data["user_profile"]["preferences"].append(preference)

    def add_fact(self, fact: str):
        """添加关于用户的事实（去重）。"""
        if fact not in self.data["user_profile"]["facts"]:
            self.data["user_profile"]["facts"].append(fact)

    def add_habit(self, habit: str):
        """添加用户习惯（去重）。"""
        if habit not in self.data["user_profile"]["habits"]:
            self.data["user_profile"]["habits"].append(habit)

    # ============ 对话摘要 ============

    def add_conversation_summary(self, summary: str, key_points: list[str] = None):
        """添加一次对话的摘要。"""
        self.data["conversation_summaries"].append({
            "date": datetime.now().isoformat(),
            "summary": summary,
            "key_points": key_points or [],
        })

    # ============ 决定 ============

    def add_decision(self, decision: str, context: str = ""):
        """记录一个重要决定。"""
        self.data["decisions"].append({
            "date": datetime.now().isoformat(),
            "decision": decision,
            "context": context,
        })

    # ============ 知识 ============

    def add_knowledge(self, topic: str, content: str):
        """添加一条知识。"""
        self.data["knowledge"].append({
            "date": datetime.now().isoformat(),
            "topic": topic,
            "content": content,
        })

    # ============ 检索 ============

    def search(self, keyword: str, limit: int = 5) -> list[dict]:
        """简单关键词检索，返回相关记忆条目。

        检索范围：用户画像、对话摘要、决定、知识。
        按匹配度排序，返回前 limit 条。
        """
        keyword = keyword.lower()
        results = []

        # 检索用户画像
        for pref in self.data["user_profile"]["preferences"]:
            if keyword in pref.lower():
                results.append({"type": "preference", "content": pref, "score": 3})
        for fact in self.data["user_profile"]["facts"]:
            if keyword in fact.lower():
                results.append({"type": "fact", "content": fact, "score": 3})
        for habit in self.data["user_profile"]["habits"]:
            if keyword in habit.lower():
                results.append({"type": "habit", "content": habit, "score": 3})

        # 检索对话摘要
        for conv in self.data["conversation_summaries"]:
            score = 0
            if keyword in conv["summary"].lower():
                score += 2
            for kp in conv["key_points"]:
                if keyword in kp.lower():
                    score += 1
            if score > 0:
                results.append({
                    "type": "conversation",
                    "content": conv["summary"],
                    "date": conv["date"],
                    "score": score,
                })

        # 检索决定
        for dec in self.data["decisions"]:
            score = 0
            if keyword in dec["decision"].lower():
                score += 2
            if keyword in dec.get("context", "").lower():
                score += 1
            if score > 0:
                results.append({
                    "type": "decision",
                    "content": dec["decision"],
                    "date": dec["date"],
                    "score": score,
                })

        # 检索知识
        for kn in self.data["knowledge"]:
            score = 0
            if keyword in kn["topic"].lower():
                score += 3
            if keyword in kn["content"].lower():
                score += 1
            if score > 0:
                results.append({
                    "type": "knowledge",
                    "content": f"[{kn['topic']}] {kn['content']}",
                    "date": kn["date"],
                    "score": score,
                })

        # 按分数排序，取前 limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # ============ 构建上下文 ============

    def build_context(self, query: str = "") -> str:
        """构建长期记忆上下文，注入到 system prompt。

        如果有 query，先检索相关内容；否则返回用户画像概要。
        """
        parts = []

        # 用户画像（始终包含）
        profile = self.data["user_profile"]
        profile_parts = []
        if profile["preferences"]:
            profile_parts.append(f"偏好：{'; '.join(profile['preferences'][-5:])}")
        if profile["facts"]:
            profile_parts.append(f"已知事实：{'; '.join(profile['facts'][-5:])}")
        if profile["habits"]:
            profile_parts.append(f"习惯：{'; '.join(profile['habits'][-5:])}")
        if profile_parts:
            parts.append("【用户画像】\n" + "\n".join(profile_parts))

        # 如果有 query，检索相关内容
        if query:
            relevant = self.search(query)
            if relevant:
                lines = ["【相关历史记忆】"]
                for r in relevant:
                    type_label = {
                        "preference": "偏好",
                        "fact": "事实",
                        "habit": "习惯",
                        "conversation": "对话",
                        "decision": "决定",
                        "knowledge": "知识",
                    }.get(r["type"], r["type"])
                    lines.append(f"- [{type_label}] {r['content']}")
                parts.append("\n".join(lines))

        return "\n\n".join(parts)

    # ============ 统计 ============

    def get_stats(self) -> dict:
        return {
            "preferences": len(self.data["user_profile"]["preferences"]),
            "facts": len(self.data["user_profile"]["facts"]),
            "habits": len(self.data["user_profile"]["habits"]),
            "conversation_summaries": len(self.data["conversation_summaries"]),
            "decisions": len(self.data["decisions"]),
            "knowledge": len(self.data["knowledge"]),
            "storage_path": self.storage_path,
        }

    def clear(self):
        """清空所有长期记忆（危险操作）。"""
        self.data = {
            "user_profile": {"preferences": [], "facts": [], "habits": []},
            "conversation_summaries": [],
            "decisions": [],
            "knowledge": [],
        }

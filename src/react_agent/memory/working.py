"""工作记忆。

当前任务的 scratchpad：
- 任务目标
- 子任务分解（待办/进行中/已完成）
- 中间结果
- 注意事项/约束

任务完成后可清空，或把重要内容转入长期记忆。
"""


class WorkingMemory:

    def __init__(self):
        self.goal: str = ""
        self.subtasks: list[dict] = []
        self.intermediate_results: list[str] = []
        self.constraints: list[str] = []
        self.task_complete: bool = False

    # ============ 目标管理 ============

    def set_goal(self, goal: str):
        self.goal = goal
        self.task_complete = False

    # ============ 子任务管理 ============

    def add_subtask(self, description: str):
        self.subtasks.append({
            "description": description,
            "status": "pending",
        })

    def start_subtask(self, description: str):
        for st in self.subtasks:
            if st["description"] == description:
                st["status"] = "in_progress"
                return

    def complete_subtask(self, description: str):
        for st in self.subtasks:
            if st["description"] == description:
                st["status"] = "done"
                return

    def get_pending_subtasks(self) -> list[str]:
        return [st["description"] for st in self.subtasks if st["status"] == "pending"]

    def get_in_progress_subtasks(self) -> list[str]:
        return [st["description"] for st in self.subtasks if st["status"] == "in_progress"]

    # ============ 中间结果 ============

    def add_result(self, result: str):
        self.intermediate_results.append(result)

    # ============ 约束管理 ============

    def add_constraint(self, constraint: str):
        if constraint not in self.constraints:
            self.constraints.append(constraint)

    # ============ 生成上下文 ============

    def build_context(self) -> str:
        parts = []

        if self.goal:
            parts.append(f"【当前任务目标】\n{self.goal}")

        if self.subtasks:
            lines = ["【子任务进度】"]
            for st in self.subtasks:
                icon = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}[st["status"]]
                lines.append(f"{icon} {st['description']}")
            parts.append("\n".join(lines))

        if self.intermediate_results:
            lines = ["【已获得的中间结果】"]
            for i, r in enumerate(self.intermediate_results[-5:], 1):
                lines.append(f"{i}. {r}")
            parts.append("\n".join(lines))

        if self.constraints:
            parts.append("【约束与注意事项】\n" + "\n".join(f"- {c}" for c in self.constraints))

        return "\n\n".join(parts)

    # ============ 任务完成 ============

    def mark_complete(self):
        self.task_complete = True

    def clear(self):
        self.goal = ""
        self.subtasks = []
        self.intermediate_results = []
        self.constraints = []
        self.task_complete = False

    def get_stats(self) -> dict:
        return {
            "has_goal": bool(self.goal),
            "subtasks_total": len(self.subtasks),
            "subtasks_done": sum(1 for st in self.subtasks if st["status"] == "done"),
            "intermediate_results": len(self.intermediate_results),
            "constraints": len(self.constraints),
            "task_complete": self.task_complete,
        }

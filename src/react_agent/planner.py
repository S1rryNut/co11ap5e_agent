"""
任务规划器。

把复杂任务拆成有序的子任务列表，再逐个执行。
Plan-and-Execute 模式的核心组件。
"""
from react_agent.llm import LLMClient


PLAN_PROMPT = """你是任务规划器。把用户的复杂任务拆成有序的子任务列表。

要求：
1. 每个子任务是一句话，描述要做什么
2. 子任务按执行顺序排列
3. 不要超过 5 个子任务
4. 只输出 JSON 数组，不要其他文字

示例：
用户：研究 AI Agent 历史并写总结
输出：["搜索 AI Agent 起源", "搜索关键里程碑", "搜索最新进展", "汇总写总结"]

用户任务：{task}
"""


class Planner:
    """任务规划器：把复杂任务拆成子任务列表。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def aplan(self, task: str) -> list[str]:
        """异步规划，返回子任务列表。"""
        prompt = PLAN_PROMPT.format(task=task)
        response = await self.llm.achat([{"role": "user", "content": prompt}])

        from react_agent.utils.json_parser import parse_json_robust
        result = parse_json_robust(response["content"])

        if isinstance(result, list):
            return [str(item) for item in result]

        # 解析失败，退化：把整个任务当一个子任务
        return [task]

    def plan(self, task: str) -> list[str]:
        """同步包装。"""
        import asyncio
        return asyncio.run(self.aplan(task))

"""
react_agent — 不依赖 LangChain 的轻量 Agent 框架

核心模块：
- LLMClient: 统一 LLM 调用，支持 DeepSeek / OpenAI
- Tool / ToolRegistry: 工具注册与自动 Schema 生成
- BaseMemory / ShortTermMemory / SummaryMemory: 记忆管理
- Agent: ReAct 核心循环
- Planner: Plan-and-Execute 规划器
"""

from react_agent.llm import LLMClient
from react_agent.tool import Tool, ToolRegistry, tool
from react_agent.memory.base import BaseMemory
from react_agent.memory.short_term import ShortTermMemory
from react_agent.memory.summary import SummaryMemory
from react_agent.agent import Agent
from react_agent.planner import Planner

__all__ = [
    "LLMClient",
    "Tool",
    "ToolRegistry",
    "tool",
    "BaseMemory",
    "ShortTermMemory",
    "SummaryMemory",
    "Agent",
    "Planner",
]

"""
自定义异常
"""


class AgentError(Exception):
    """Agent 基础异常"""


class LLMRateLimitError(AgentError):
    """LLM 限流（429）"""


class LLMTimeoutError(AgentError):
    """LLM 调用超时"""


class LLMAPIError(AgentError):
    """LLM API 其他错误"""


class ToolNotFoundError(AgentError):
    """工具不存在"""


class ToolArgumentError(AgentError):
    """工具参数解析失败"""


class ToolExecutionError(AgentError):
    """工具执行异常"""


class ToolTimeoutError(AgentError):
    """工具执行超时"""


class MaxIterationsError(AgentError):
    """达到最大迭代次数"""


class ConsecutiveFailureError(AgentError):
    """连续失败次数过多"""

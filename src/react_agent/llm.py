import asyncio
from openai import AsyncOpenAI
from openai import RateLimitError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.max_retries = max_retries

        # AsyncOpenAI 初始化只接受 api_key / base_url / timeout
        # model 是在调用 chat.completions.create 时才传的
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        reraise=True,
    )
    async def _call_with_retry(self, **kwargs):
        """带重试的 API 调用。只重试 429 和 5xx，其他错误直接抛。"""
        return await self._client.chat.completions.create(**kwargs)

    async def achat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> dict:
        # 组装请求参数
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # 调用（带重试）
        response = await self._call_with_retry(**kwargs)

        # 提取第一条选择
        choice = response.choices[0]
        message = choice.message

        # 统一成我们自己的格式
        result = {
            "content": message.content or "",
            "tool_calls": None,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

        # 如果有工具调用，提取出来（保持 OpenAI 原始格式：type + function 嵌套）
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return result

    async def astream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        stream = await self._client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                yield {
                    "type": "text",
                    "data": {"delta": delta.content},
                }

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield {
                        "type": "tool_call",
                        "data": {
                            "index": tc.index,
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function else "",
                            "arguments_delta": tc.function.arguments if tc.function else "",
                        },
                    }

            if finish_reason:
                yield {
                    "type": "done",
                    "data": {"finish_reason": finish_reason},
                }

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> dict:
        """同步包装。只能在脚本里用，不能在已有事件循环的环境调用。"""
        return asyncio.run(self.achat(messages, tools, temperature, max_tokens))

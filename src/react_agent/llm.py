import asyncio
from openai import AsyncOpenAI
from openai import RateLimitError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url:str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.max_retries = max_retries

        # 创建异步客户端
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
        # 带重试的 API 调用。只重试 429 和 5xx，其他错误直接抛。
        return await self._client.chat.completions.create(**kwargs)

    async def achat(self, messages, tools = None, temperature = 0.7, max_tokens = None):
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

        # 调用 API
        response = await self._call_with_retry(**kwargs)

        # 提取第一条选择
        choice = response.choices[0]
        message = choice.message

        result = {
            "content": message.content or "",
            "tool_calls": None,
            "finish_reason": choice.finish_reason,
            "usage":{
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
        }
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
        # 组装请求参数
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

        # 流式调用
        stream = await self._client.chat.completions.create(**kwargs)

        # 遍历每个 chunk
        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # 情况 1：有文本内容
            if delta.content:
                yield {
                    "type": "content",
                    "data": {"delta": delta.content},
                }

            # 情况 2：有工具调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield {
                        "type": "tool_call",
                        "data": {
                            "index": tc.index,
                            "id": tc.id or None,
                            "name": tc.function.name if tc.function else None,
                            "arguments": tc.function.arguments if tc.function else None,
                        },
                    }
            # 情况 3：完成原因
            if finish_reason:
                yield {
                    "type": "finish_reason",
                    "data": {"finish_reason": finish_reason},
                }

    def chat(self, messages, tools = None, temperature = 0.7, max_tokens = None):
        # 同步调用异步方法
        return asyncio.run(self.achat(messages, tools, temperature, max_tokens))

# co11ap5e_agent

不依赖 LangChain 的轻量 ReAct Agent 框架。用原生 OpenAI SDK 从零构建，兼容 DeepSeek / DashScope / OpenAI。

## 特性

- **零框架依赖**：只依赖 openai / pydantic / httpx，不引入 LangChain 等重型框架
- **ReAct 核心循环**：思考 → 调工具 → 看结果 → 再思考，直到完成
- **自动工具 Schema**：`@tool` 装饰器从函数签名自动生成 OpenAI Function Calling JSON Schema
- **多种记忆策略**：滑动窗口（ShortTermMemory）、摘要压缩（SummaryMemory）
- **容错 JSON 解析**：处理 LLM 输出的不规范 JSON（代码块、注释、尾随逗号、单引号）
- **Plan-and-Execute**：内置规划器，复杂任务先拆解再执行
- **生产级错误处理**：工具超时、连续失败、最大迭代次数保护
- **流式输出支持**：`astream_chat` 支持逐 token 输出

## 安装

```bash
# 克隆仓库
git clone https://github.com/S1rryNut/co11ap5e_agent.git
cd co11ap5e_agent

# 用 uv 安装（推荐）
uv sync

# 或用 pip
pip install -e .
```

## 快速开始

### 1. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API Key：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxx
DASHSCOPE_API_KEY=sk-xxxxxxxx
```

### 2. 跑 Demo

```bash
uv run python examples/demo.py
```

### 3. 最小示例

```python
import asyncio
import os
from dotenv import load_dotenv
from react_agent import LLMClient, Agent, tool

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，比如 '123 * 456'。"""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """查询某城市的天气。"""
    return f"{city}：晴 25℃"

llm = LLMClient(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

agent = Agent(
    llm=llm,
    tools=[calculator, get_weather],
    system_prompt="你是一个助手，需要计算用 calculator，需要天气用 get_weather。",
    verbose=True,
)

async def main():
    answer = await agent.arun("12345 乘以 67890 等于多少？苏州天气怎么样？")
    print("最终回答:", answer)

asyncio.run(main())
```

## 架构

```
react_agent/
├── llm.py            # LLM 客户端封装（统一调用格式 + 重试 + 流式）
├── tool.py           # 工具系统（@tool 装饰器 + 自动 Schema 生成 + 注册表）
├── agent.py          # ReAct 核心循环
├── planner.py        # 任务规划器（Plan-and-Execute）
├── errors.py         # 自定义异常
├── memory/
│   ├── base.py       # 记忆抽象基类
│   ├── short_term.py # 滑动窗口记忆
│   └── summary.py    # 摘要压缩记忆
└── utils/
    ├── token_counter.py  # Token 计数
    └── json_parser.py    # 容错 JSON 解析
```

## 核心概念

### ReAct 循环

Agent 解决问题的过程是一个循环：

```
用户提问
  ↓
LLM 思考：要不要调工具？
  ↓ 调工具
工具执行 → 返回结果
  ↓
LLM 再思考：信息够了吗？
  ↓ 够了
输出最终答案
```

### 工具系统

用 `@tool` 装饰器把普通函数变成 LLM 可调用的工具：

```python
@tool
def search(query: str, max_results: int = 3) -> str:
    """搜索互联网。"""
    # ...
```

框架自动从类型注解和 docstring 生成 JSON Schema，不需要手写。

### 记忆系统

- `ShortTermMemory`：滑动窗口，保留最近的消息，总 token 不超过上限
- `SummaryMemory`：超过阈值时把旧消息压缩成摘要，保留关键信息

## 开发

### 运行测试

```bash
uv run pytest tests/ -v
```

### 项目结构

```
Agent/
├── src/react_agent/   # 框架源码
├── examples/          # 示例代码
├── tests/             # 单元测试
├── pyproject.toml     # 项目配置
└── README.md
```

## 兼容的 LLM

| 服务商 | base_url | model 示例 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| 阿里云 DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| OpenAI | （默认） | `gpt-4o` |

只要兼容 OpenAI API 格式的服务都能用。

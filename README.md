# co11ap5e_agent

不依赖 LangChain 的轻量 ReAct Agent 框架。用原生 OpenAI SDK 从零构建，兼容 DeepSeek / DashScope / OpenAI。

## 特性

- **零框架依赖**：只依赖 openai / pydantic / httpx，不引入 LangChain 等重型框架
- **ReAct 核心循环**：思考 → 调工具 → 看结果 → 再思考，直到完成
- **自动工具 Schema**：`@tool` 装饰器从函数签名自动生成 OpenAI Function Calling JSON Schema
- **三层记忆体系**：
  - **短期记忆**：原始消息 + 摘要 + 结构化记忆，动态阈值自动压缩
  - **工作记忆**：当前任务 scratchpad（目标/子任务/中间结果/约束），随任务清空
  - **长期记忆**：JSON 文件持久化，跨会话保留用户画像/对话摘要/知识，支持关键词检索
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
├── agent.py          # ReAct 核心循环（含记忆自动压缩）
├── planner.py        # 任务规划器（Plan-and-Execute）
├── errors.py         # 自定义异常
├── memory/
│   ├── base.py       # 记忆抽象基类
│   ├── short_term.py # 短期记忆（原始消息+摘要+结构化记忆，自动压缩）
│   ├── working.py    # 工作记忆（当前任务 scratchpad：目标/子任务/中间结果/约束）
│   └── long_term.py  # 长期记忆（JSON 持久化，跨会话，关键词检索）
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

三层记忆协同工作：

```
用户输入
  → 长期记忆检索相关内容 + 工作记忆上下文 → 注入 system prompt
  → 短期记忆维护对话历史（自动压缩）
  → 工具结果 → 记录到工作记忆
  → 任务完成 → 摘要写入长期记忆并持久化
```

**短期记忆 `ShortTermMemory`**：三层分层，自动压缩
- 原始消息：最近 3 轮保留原文
- 摘要记忆：更早的对话经 LLM 摘要，渐进式追加
- 结构化记忆：facts / preferences / decisions / open_questions
- 动态压缩：token 达上限 70% 时触发，压缩到 35%

**工作记忆 `WorkingMemory`**：当前任务的 scratchpad
- 任务目标、子任务进度（待办/进行中/已完成）
- 工具调用中间结果（自动记录，保留最近 5 条）
- 约束与注意事项（自动去重）
- 任务完成后清空

**长期记忆 `LongTermMemory`**：跨会话持久化
- JSON 文件存储，启动时自动加载
- 用户画像（偏好/事实/习惯）、对话摘要、重要决定、知识条目
- 关键词检索（按匹配度排序），自动注入相关历史到 system prompt
- 损坏文件容错

```python
from react_agent.memory import ShortTermMemory, WorkingMemory, LongTermMemory

agent = Agent(
    llm=llm,
    tools=tools,
    memory=ShortTermMemory(max_tokens=8000, system_prompt="..."),
    working_memory=WorkingMemory(),
    long_term_memory=LongTermMemory(storage_path="my_memory.json"),
)
```

不传 `working_memory` / `long_term_memory` 时退化为纯短期记忆，完全向后兼容。

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

## 免责声明与使用须知

> ⚠️ **请在使用前阅读本节**

### 项目性质

本项目是**教学/学习项目**，用于理解 AI Agent 的底层实现原理（ReAct 循环、工具系统、记忆系统）。它不是商业级产品，**不建议用于生产环境、重要代码库或无法承受错误的场景**。

### 能力边界

| 项目 | 现状 |
|---|---|
| 模型支持 | 单模型驱动（DeepSeek / DashScope / OpenAI 可选），无自动故障切换 |
| 上下文管理 | 有记忆系统，但上下文窗口有限，超长对话可能丢失细节 |
| 规划能力 | 基于 LLM 自主决策，无确定性规划器兜底 |
| 并发能力 | 单循环串行执行，无并行子任务 |
| 工具范围 | 文件读写、Shell 命令、代码搜索、Git 操作等内置工具，无 web 搜索、截图等 |
| 交互方式 | 命令行，无 IDE 集成 |

### 使用风险（重要）

1. **工具会直接修改你的文件**：`write_file` / `edit_file` 会真实写入文件，`run_shell_command` 会真实执行命令。**没有 diff 预览、没有撤销机制、没有用户确认流程**。请在临时目录/测试项目中运行，或自行备份。
2. **LLM 可能出错**：模型可能误读需求、写错代码、误用工具参数。所有输出都应人工审查后再使用。
3. **API 成本**：长时间任务会消耗较多 token，请注意 API 额度。
4. **密钥安全**：`.env` 文件（含 API Key）已被 `.gitignore` 排除，请勿手动提交。

### 其他说明

- 本项目参考了主流 Agent 框架的设计思想，但实现完全独立。
- 如果你发现了 bug 或有改进建议，欢迎提交 Issue 或 PR。

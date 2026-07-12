"""数据模型定义。


核心概念：
- tools：小杉传来的工具列表（机器人能做的动作，如挥手、走路）
- tool_calls：大模型返回的工具调用决策（决定让机器人做什么动作）
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================
# 请求相关模型
# ============================================================


#定义一个名为 ChatMessage 的类，它继承自 BaseModel
#BaseModel 来自 Pydantic 库（第三方库，需要 pip install pydantic）。继承它之后，你的ChatMessage 类会自动获得以下超能力：
#1.自动类型校验 2. 自动序列化/反序列化 3.自动补全默认值
class ChatMessage(BaseModel):
    """一条对话消息。

    OpenAI 接口中，messages 是一个列表，每个元素是一条消息。
    消息有不同的角色（role）：

    - system：系统提示词，告诉大模型它是什么角色、该怎么回答
    - user：用户说的话
    - assistant：大模型之前回复的内容
    - tool：工具执行的结果（比如机器人挥手后返回的状态）

    MCP 工具调用流程中的消息变化：
    1. 小杉发送：system + user 消息（加上 tools 工具列表）
    2. 大模型返回：assistant 消息（可能包含 tool_calls，表示"让机器人挥手"）
    3. 小杉执行工具后发送：assistant + tool 消息（tool 消息包含执行结果）
    4. 大模型继续返回：assistant 消息（最终的文字回复）
    """

    role: Literal['system', 'user', 'assistant', 'tool']
    content: str | None = None

    # 当 assistant 消息包含工具调用时，这里填充调用详情
    # 格式：[{"id": "call_xxx", "type": "function", "function": {"name": "self.otto.hand_wave", "arguments": "{}"}}]
    tool_calls: list[dict[str, Any]] | None = None

    # 当 role='tool' 时，标记是对哪次工具调用的回复
    tool_call_id: str | None = None


class ChatCompletionsRequest(BaseModel):
    """OpenAI Chat Completions 兼容的请求体。

    这是小杉机器人发来的请求的完整结构。

    重点字段：
    - tools：小杉传来的机器人 MCP 工具列表
      示例：[{"type": "function", "function": {"name": "self.otto.hand_wave", ...}}]
    - stream：是否使用 SSE 流式返回（小杉默认用流式）
    """

    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.7
    stream: bool = False

    # 小杉传入的工具列表（OpenAI tools 格式）
    # 每个工具描述了机器人可以做的一个动作
    tools: list[dict[str, Any]] | None = None

    # 工具选择策略（小杉一般用默认的 "auto"）
    tool_choice: dict[str, Any] | str | None = None


# ============================================================
# SSE 流式响应相关模型
# ============================================================


class ChatCompletionChunkDelta(BaseModel):
    """流式响应中每个增量片段的内容。



    MCP 场景下的增量：
    - content：大模型的文字回复片段
    - tool_calls：工具调用增量（可能分多个 chunk 发送，小杉会按 index 拼接）
    """

    role: Literal['assistant'] | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChunkChoice(BaseModel):
    """流式响应中的一个选项。"""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Literal['stop', 'tool_calls'] | None = None


class ChatCompletionsChunk(BaseModel):
    """单个 SSE 数据块对应的 JSON 结构。"""

    id: str = Field(default_factory=lambda: f'chatcmpl-{uuid.uuid4().hex}')
    object: Literal['chat.completion.chunk'] = 'chat.completion.chunk'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


def to_sse_data(payload: ChatCompletionsChunk | str) -> str:
    """把 Python 对象或字符串转换成 SSE 格式文本。

    SSE 的基本格式：data: JSON内容\\n\\n

    最后一个 [DONE] 是小杉识别流结束的标记。
    """

    if isinstance(payload, str):
        return f'data: {payload}\n\n'
    return f'data: {json.dumps(payload.model_dump(exclude_none=True), ensure_ascii=False)}\n\n'

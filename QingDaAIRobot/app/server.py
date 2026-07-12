"""FastAPI 服务端。

教学说明：
- 这个文件是整个项目的"前台"。
- 它对外提供一个 HTTP 接口：POST /v1/chat/completions
- 小杉机器人就通过这个接口和我们对话。

核心逻辑：
1. 收到小杉的请求（包含对话消息 + 机器人 MCP 工具列表）
2. 把请求转发给智能体服务（agent.py）
3. 智能体调用百炼大模型
4. 把大模型的回复（文字 + 工具调用）返回给小杉

如果大模型决定让机器人做动作（如挥手），
返回结果中会包含 tool_calls，小杉收到后会执行对应的 MCP 工具。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.agent import XiaozhiAgentService
from app.config import get_settings
from app.models import (
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionsChunk,
    ChatCompletionsRequest,
    to_sse_data,
)

logger = logging.getLogger('uvicorn.error')


# ============================================================
# FastAPI 应用初始化
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。

    教学说明：
    - 应用启动时执行这里。
    - 我们在这里创建智能体服务实例，挂在 app.state 上全局共享。
    - 这样每次请求不需要重新创建，提高性能。
    """
    settings = get_settings()
    app.state.agent = XiaozhiAgentService(settings)
    yield


app = FastAPI(title='小杉机器人 LangChain 智能体', version='1.0.0', lifespan=lifespan)


# ============================================================
# 健康检查接口
# ============================================================


@app.get('/health')
async def health() -> dict[str, str]:
    """健康检查接口。

    作用：快速确认服务有没有成功启动。
    """
    return {'status': 'ok'}


# ============================================================
# SSE 流式处理函数
# ============================================================


async def stream_response(request: ChatCompletionsRequest):
    """SSE 流式返回大模型的回复。

    教学说明：
    - 这个函数负责把智能体的输出按 SSE 格式逐块发给小杉。
    - 每个 SSE 块都是标准 OpenAI 格式的 JSON。
    - 如果大模型决定调用工具（让机器人做动作），
      SSE 块中的 delta.tool_calls 会包含工具调用信息。
    """

    model_name = request.model or get_settings().dashscope_model

    # 首包：声明 assistant 角色（OpenAI 标准要求）
    first_chunk = ChatCompletionsChunk(
        model=model_name,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role='assistant'),
            )
        ],
    )
    chunk_id = first_chunk.id
    created = first_chunk.created
    yield to_sse_data(first_chunk)

    # 标记是否出现了工具调用
    has_tool_calls = False

    # 逐块获取大模型输出
    async for content, tool_calls in app.state.agent.astream_with_tools(
        request.messages, request.tools
    ):
        delta = ChatCompletionChunkDelta()

        # 文字增量
        if content:
            delta.content = content

        # 工具调用增量
        if tool_calls:
            has_tool_calls = True
            delta.tool_calls = tool_calls

        # 只有 delta 有内容时才发送
        if delta.content or delta.tool_calls:
            yield to_sse_data(
                ChatCompletionsChunk(
                    id=chunk_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=delta,
                        )
                    ],
                )
            )

    # 结束块
    # 有工具调用 → finish_reason='tool_calls'
    # 纯文字回复 → finish_reason='stop'
    finish_reason = 'tool_calls' if has_tool_calls else 'stop'
    yield to_sse_data(
        ChatCompletionsChunk(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason=finish_reason,
                )
            ],
        )
    )
    yield to_sse_data('[DONE]')


# ============================================================
# 主接口：/v1/chat/completions
# ============================================================


@app.post('/v1/chat/completions')
async def create_chat_completion(request: ChatCompletionsRequest):
    """兼容 OpenAI Chat Completions 的主接口。

    这是小杉机器人调用我们的唯一入口。

    请求示例（小杉发送的）：
    {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "你好"}
        ],
        "stream": true,
        "tools": [
            {"type": "function", "function": {"name": "self.otto.hand_wave", ...}}
        ]
    }

    返回（流式时）：
    - SSE 流，每块包含文字增量或工具调用增量
    - 如果大模型决定让机器人挥手，流中会包含 tool_calls

    返回（非流式时）：
    - 完整的 JSON，包含最终文字和工具调用
    """

    logger.info(f'收到请求 time={datetime.now().isoformat()}')

    # 流式模式
    if request.stream:
        return StreamingResponse(
            stream_response(request),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            },
        )

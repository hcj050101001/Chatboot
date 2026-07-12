"""智能体服务核心模块。

工作流程：
  小杉发来消息(含 tools) → 我们的智能体 → 百炼大模型 → 返回 tool_calls
                                                                    ↓
  小杉执行动作(挥手/走路) ← 小杉收到 tool_calls ← 我们的智能体 ←←←←←←

为什么不在智能体里直接控制机器人？
  因为机器人连接在小杉服务器上，我们的智能体是独立运行的 HTTP 服务，
  无法直接访问机器人硬件。只能通过 tool_calls "通知"小杉去执行。
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.models import ChatMessage

logger = logging.getLogger('xiaozhi.agent')


# ============================================================
# 系统提示词（System Prompt）
# ============================================================
# 这是整个项目最重要的部分之一！
# 系统提示词决定了大模型"认为自己是干什么的"、"该怎么回答"。
#
# 我们在这里告诉大模型：
# 1. 它是一个小杉机器人的"大脑"
# 2. 它可以控制机器人的身体动作（通过 MCP 工具）
# 3. 在什么场景下应该主动让机器人做动作
#
# 学生可以自由修改这个提示词，创造不同性格的机器人！
# ============================================================

SYSTEM_PROMPT = """你是一个搭载在 Otto 双足机器人上的 AI 智能体，你的名字叫"小杉"。
你不仅能用语言和人交流，还能控制机器人的身体做各种动作。

## 你的身体能力
你通过 MCP（Model Context Protocol）工具来控制机器人的身体。
工具列表会由系统在每次对话时动态提供，请查看 tools 参数中的工具描述来了解具体可用的动作和参数。

你通常可以做这些类型的动作：
- 挥手：打招呼、告别、表达友善
- 行走：向前走、后退
- 转身：左转、右转
- 跳跃：开心时跳跃
- 摇摆：左右摇摆，表示愉快或跳舞
- 太空步：炫酷的太空步
- 弯曲：弯腰鞠躬
- 举手：举手抢答、表示兴奋
- 其他动作：具体取决于当前固件提供的工具

## 行为准则
1. **主动配合动作**：当用户和你打招呼时，你应该在说话的同时调用动作工具让机器人挥手；当用户说"过来"时，你应该让机器人走过来；当用户让你跳舞时，调用摇摆或跳跃工具。
2. **语言简短有力**：你的语音回复会被转成语音播放，所以要简短、口语化，每次回复控制在 1-2 句话。
3. **先行动后说话**：如果需要做动作，先调用工具（让机器人动起来），然后再输出文字。
4. **安全第一**：不要让机器人做危险动作，行走步数不要太多（一般 3-10 步）。

## 重要提示
- 每次对话时请仔细查看 tools 列表中的工具名称和参数描述，不同固件版本的工具名可能不同。
- 不要凭记忆猜测工具名，必须根据 tools 列表中实际提供的工具来调用。
- 如果用户要求做一个动作，请在 tools 列表中找到最匹配的工具并调用它。

## 对话风格
- 热情、活泼、有亲和力
- 像一个开朗的小伙伴
- 用中文交流
- 可以适当使用语气词（"嘿"、"哇"、"好的呀"等）

## 示例
用户："你好呀！"
你应该：在 tools 中找到挥手相关的工具并调用，同时回复"你好呀！很高兴见到你！"

用户："过来一下"
你应该：在 tools 中找到行走相关的工具并调用，同时回复"好的，我来啦！"

用户："跳个舞吧"
你应该：在 tools 中找到摇摆或跳跃相关的工具并调用，然后回复"没问题，看我的！"

用户："你的电量还够吗？"
你应该：在 tools 中找到电池查询相关的工具并调用，根据结果回复。
""".strip()


class XiaozhiAgentService:
    """小杉机器人智能体服务。

    说明：
    - 这个类的核心职责是：把小杉发来的请求"转发"给阿里百炼大模型。
    - 它不做任何工具执行（不直接控制机器人）。
    - 它只是"搬运工"：把 tools 搬给大模型，把 tool_calls 搬回给小杉。
    - 真正的机器人控制由小杉在设备端完成。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # 创建两个 LLM 实例：
        # 一个用于非流式调用，一个用于流式调用
        # 复用实例避免每次请求都创建新的连接
        self.llm = self._create_llm(streaming=False)
        self.llm_streaming = self._create_llm(streaming=True)

    def _create_llm(self, *, streaming: bool) -> ChatOpenAI:
        """创建底层大模型客户端。

        教学说明：
        - langchain_openai.ChatOpenAI 是一个兼容 OpenAI 接口的聊天模型客户端。
        - 阿里百炼提供了兼容 OpenAI 的 API，所以我们可以直接用它。
        - 参数含义：
            - api_key：百炼平台的 API Key
            - base_url：百炼的 OpenAI 兼容接口地址
            - model：使用哪个模型（qwen-plus 性价比最高）
            - streaming：是否启用流式输出
        """
        return ChatOpenAI(
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_base_url,
            model=self.settings.dashscope_model,
            temperature=0.7,
            streaming=streaming,
        )

    # ============================================================
    # MCP 工具调用 - 核心方法
    # ============================================================

    async def astream_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[tuple[str | None, list[dict[str, Any]] | None], None]:
        """MCP 代理模式 - 流式：逐块返回 (content, tool_calls)。

        教学说明（最重要的方法！）：
        - 小杉默认使用 SSE 流式模式调用我们的接口。
        - 在流式模式下，大模型的输出是逐块返回的：
          - 有些块只包含文字（content）
          - 有些块包含工具调用（tool_calls）
        - 我们需要把每种类型的块都正确地返回给小杉。

        参数：
            messages: 对话历史
            tools: 机器人可用的 MCP 工具列表

        每次 yield：
            (content, tool_calls) 元组
            - content: 文字片段（可能为 None）
            - tool_calls: 工具调用增量（可能为 None）
        """

        # 1. 转换消息格式
        lc_messages = self._convert_messages(messages)

        # 2. 绑定工具并创建流式调用
        # 注意：小杉在某些场景（如意图识别）可能不传 tools，此时走纯文本模式
        llm_with_tools = self.llm_streaming
        if tools:
            # 打印小杉传来的 MCP 工具列表，方便学生观察
            tool_names = [
                t.get('function', {}).get('name', '未知')
                if isinstance(t, dict)
                else getattr(getattr(t, 'function', None), 'name', '未知')
                for t in tools
            ]
            logger.info(f'📋 小杉传来 {len(tools)} 个 MCP 工具: {tool_names}')
            llm_with_tools = self.llm_streaming.bind_tools(tools)
        else:
            logger.info('📋 本次请求未携带 MCP 工具，走纯文本模式')

        # 打印用户最新说了什么
        if messages:
            last_user_msg = next(
                (m for m in reversed(messages) if m.role == 'user'), None
            )
            if last_user_msg:
                logger.info(f'👤 用户说: {last_user_msg.content}')

        # 3. 流式获取大模型输出
        # 用于收集本次流中所有被调用的工具名称（工具名在第一个 chunk 中出现）
        called_tools: set[str] = set()

        async for chunk in llm_with_tools.astream(lc_messages):
            content = None
            tool_calls = None

            # 提取文字内容增量
            if hasattr(chunk, 'content') and chunk.content:
                if isinstance(chunk.content, str):
                    content = chunk.content
                elif isinstance(chunk.content, list):
                    # 某些模型返回内容块列表
                    texts = []
                    for item in chunk.content:
                        text = getattr(item, 'text', None)
                        if text:
                            texts.append(text)
                        elif isinstance(item, dict) and item.get('text'):
                            texts.append(item['text'])
                    if texts:
                        content = ''.join(texts)

            # 提取工具调用增量
            # tool_call_chunks 是流式增量的工具调用
            # 大模型会把一个完整的 tool_calls 拆成多个小块发送
            # 小杉端会自动按 index 把它们拼起来
            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                tool_calls = []
                for i, tc_chunk in enumerate(chunk.tool_call_chunks):
                    # 每个 chunk 可能只包含部分信息（比如只有 name 或只有 args 的一部分）
                    chunk_dict = tc_chunk if isinstance(tc_chunk, dict) else {}
                    # 记录被调用的工具名称（只在 name 非空时才打印）
                    chunk_name = chunk_dict.get('name', '')
                    if chunk_name:
                        called_tools.add(chunk_name)
                    tool_calls.append(
                        {
                            'index': chunk_dict.get('index', i),
                            'id': chunk_dict.get('id', ''),
                            'type': 'function',
                            'function': {
                                'name': chunk_name,
                                'arguments': chunk_dict.get('args', ''),
                            },
                        }
                    )

            # 只有当 content 或 tool_calls 有值时才 yield
            if content or tool_calls:
                print(content)
                yield content, tool_calls

        # 流结束后，打印本次调用的工具汇总
        if called_tools:
            logger.info(f'🤖 本次调用的 MCP 工具: {list(called_tools)}')
        elif tools:
            logger.info('🤖 本次未调用任何 MCP 工具，纯文本回复')

    # ============================================================
    # 辅助方法
    # ============================================================

    def _convert_messages(self, messages: list[ChatMessage]) -> list[BaseMessage]:
        """把 ChatMessage 列表转换为 LangChain 消息对象。

        教学说明：
        - 小杉发来的是 OpenAI 格式的消息（dict 或 Pydantic 模型）。
        - LangChain 需要自己的消息类型（SystemMessage、HumanMessage 等）。
        - 这个方法负责做格式转换。

        需要处理的消息类型：
        - system → SystemMessage（系统提示词）
        - user → HumanMessage（用户说的话）
        - assistant → AIMessage（大模型之前的回复，可能包含 tool_calls）
        - tool → ToolMessage（工具执行的结果）
        """
        result: list[BaseMessage] = []
        system_prompt_injected = False
        for msg in messages:
            if msg.role == 'system':
                # 把我们的 SYSTEM_PROMPT 追加到小杉自带的系统提示词后面
                # 这样大模型就能同时收到小杉的角色设定和我们的动作引导指令
                combined_content = (msg.content or '') + '\n\n' + SYSTEM_PROMPT
                result.append(SystemMessage(content=combined_content))
                system_prompt_injected = True
            elif msg.role == 'assistant':
                ai_msg = AIMessage(content=msg.content or '')
                if msg.tool_calls:
                    ai_msg.additional_kwargs['tool_calls'] = msg.tool_calls
                result.append(ai_msg)
            elif msg.role == 'tool':
                result.append(
                    ToolMessage(
                        content=msg.content or '',
                        tool_call_id=msg.tool_call_id or '',
                    )
                )
            else:  # user
                result.append(HumanMessage(content=msg.content or ''))
        # 如果小杉没有发 system 消息，手动插入我们的系统提示词
        if not system_prompt_injected:
            result.insert(0, SystemMessage(content=SYSTEM_PROMPT))
        return result

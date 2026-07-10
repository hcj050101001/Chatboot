"""
对话小助手 - API服务
"""
import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles

#导入RagAssistant核心类
if __package__:
    from .rag_chat import DOCS_DIR, VECTOR_DB_PATH, get_assistant
else:
    from rag_chat import DOCS_DIR, VECTOR_DB_PATH, get_assistant

#静态文件地址
STATIC_DIR=Path(__file__).resolve().parent / "static"

logger=logging.getLogger(__name__)

#创建FastAPI应用对象
app=FastAPI(title="对话小助手",description="基于知识库的智能问答助手")

#挂在静态文件目录
app.mount("/static",StaticFiles(directory=STATIC_DIR),name="static")

#当前登录用户
current_user=None

#用户会话保存的持久化文件名称
STORAGE_FILE="api_rag_session.json"

def sse_message(content, done=False):
    """构造一条 SSE 数据帧。"""
    data=json.dumps({"content":content,"done":done},ensure_ascii=False)
    return f"data: {data}\n\n"


def complete_message(content):
    """发送单条消息和明确的流结束信号。"""
    yield sse_message(content)
    yield sse_message("",done=True)


@app.post("/chat")
async def chat_stream(request:Request):
    """接受页面请求并响应流式输出内容接口"""
    global current_user

    try:
        body=await request.json()
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400,detail="请求体必须是合法 JSON") from error

    if not isinstance(body,dict):
        raise HTTPException(status_code=400,detail="请求体必须是 JSON 对象")

    question=body.get("question","")
    if not isinstance(question,str):
        raise HTTPException(status_code=400,detail="question 必须是字符串")
    question=question.strip()
    if not question:
        raise HTTPException(status_code=400,detail="问题不能为空")

    if not current_user:
        if question.startswith("/login"):
            current_user=question[7:].strip()
            content=f"已经登录为：{current_user}" if current_user else "请先登录"
            return StreamingResponse(complete_message(content),media_type="text/event-stream")
        return StreamingResponse(complete_message("请先登录"),media_type="text/event-stream")

    try:
        assistant=get_assistant(STORAGE_FILE)
    except Exception as error:
        logger.exception("初始化问答助手失败")
        raise HTTPException(status_code=503,detail="问答服务暂时不可用") from error

    def generation():
        chunk_count=0
        try:
            for chunk in assistant.chat_stream(question,current_user):
                if chunk is None:
                    continue
                chunk_count+=1
                yield sse_message(str(chunk))
        except Exception:
            logger.exception("流式回答失败")
            yield sse_message("回答生成失败，请稍后重试。")
        logger.info("流式输出结束，共输出 %s 个片段",chunk_count)
        yield sse_message("",done=True)

    response=StreamingResponse(
        generation(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":"no-cache", #不进行缓存，将内容直接在页面展示
            "Connection":"keep-alive", #告知客户端和服务器一直保持连，一个连接传输多个响应
            "X-Accel-Buffering":"no" #针对于反省代服务器的nignx，不要缓冲这个响应
        }
    )
    return response


@app.get("/")
async def root():
    """访问根路径重定向到聊天页面"""
    return RedirectResponse(url="/static/chat.html")

if __name__ == "__main__":
    print("="*50)
    print("启动智能问答小助手服务")
    print("="*50)
    print(f"知识库所在位置：{DOCS_DIR}")
    print(f"向量文件保存位置:{VECTOR_DB_PATH}")
    print("接口地址：POST http://localhost:8001/chat，JSON 请求体：{\"question\": \"问题\"}")
    print("页面地址1：http://localhost:8001/")
    print("页面地址2：http://localhost:8001/static/chat.html")

    uvicorn.run(app,host="0.0.0.0",port=8001)

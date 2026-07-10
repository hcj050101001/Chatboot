"""
对话小助手 - API服务
"""
import json

import uvicorn
from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse,RedirectResponse

#导入RagAssistant核心类
from rag_chat import get_assistant,DOCS_DIR,VECTOR_DB_PATH

#创建FastAPI应用对象
app=FastAPI(title="对话小助手",description="基于知识库的智能问答助手")

@app.get("/chat")
async def chat_stream(request:Request):
    """接受页面请求并响应流式输出内容接口"""

    try:
        #get请求获取参数
        question=request.query_params.get("questions").strip()

        if not question:
            return {"error":"问题不能为空"}

        assistant=get_assistant()

        def generation():
            chunk_count=0
            for chunk in assistant.chat_stream(question,"铁蛋"):
                chunk_count+=1
                data=json.dumps({"content":chunk,"done":False},ensure_ascii=False)
                yield f"data: {data}\n\n"

            #发送完成信号
            done_data = json.dumps({"content": chunk, "done": False}, ensure_ascii=False)
            yield f"data: {done_data}\n\n"

            print(f"流式输出完成，共输出{chunk}个片段")

        return StreamingResponse(
            generation(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":"no-cache", #不进行缓存，将内容直接在页面展示
                "Connection":"keep-alive", #告知客户端和服务器一直保持连，一个连接传输多个响应
                "X-Accel-Buffering":"no" #针对于反省代服务器的nignx，不要缓冲这个响应
            }
        )

    except Exception as e:
        return {"error":f"请求处理失败:{str(e)}"}

if __name__ == "__main__":
    print("="*50)
    print("启动智能问答小助手服务")
    print("="*50)
    print(f"知识库所在位置：{DOCS_DIR}")
    print(f"向量文件保存位置:{VECTOR_DB_PATH}")
    print("请求地址：http://localhost:8001/chat?questions=问题")

    uvicorn.run(app,host="0.0.0.0",port=8001)

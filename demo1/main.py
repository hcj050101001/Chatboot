"""项目统一启动入口：在 PyCharm 中直接运行本文件即可启动网页。"""
import os
import threading
import webbrowser

import uvicorn

from ai_chat.llm_client import get_llm_model
from setup_database import initialize_database


def main():
    """主函数"""
    try:
        model = get_llm_model()
    except RuntimeError as error:
        print(f"错误：{error}")
        print("请确保 .env 包含 LLM_API_KEY、LLM_BASE_URL 和 LLM_MODEL")
        return

    print(f"当前对话模型：{model}")
    if not initialize_database():
        print("提示：ERP 数据库暂不可用；知识库问答和其他功能仍可启动。")

    url="http://127.0.0.1:8001/"
    print(f"智能小助手页面：{url}")
    if os.getenv("OPEN_BROWSER","1") != "0":
        threading.Timer(1.5,lambda:webbrowser.open(url)).start()
    uvicorn.run("ai_chat.api:app",host="127.0.0.1",port=8001,reload=False)


if __name__ == "__main__":
    main()

import os
import dashscope
from dashscope import Generation
from dotenv import load_dotenv

load_dotenv()

dashscope.api_key = os.getenv("API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

#设置System_Message
SYSTEM_MESSAGE="你是一个公司的HR"

# 定义列表保存对话历史，并作为参数进行传递
chat_history = []

def add_chart_history(role,content):
    """添加对话到历史消息"""
    chat_history.append(
        {"role": role, "content": content}
    )


def _extract_chunk_content(response):
    """安全提取流式响应里的文本片段。"""
    output = getattr(response, "output", None)
    choices = getattr(output, "choices", None)
    if not choices:
        return None

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if not message:
        return None

    return getattr(message, "content", None)


#调用方法将系统提示词添加到对话历史
add_chart_history("system",SYSTEM_MESSAGE)

def chat(prompt, model=None):
    """
    对话函数
    Args：
        prompt：用户提问的问题
        model：模型名称，默认使用.env中的配置
    Returns：
        模型回复的文本信息
    """
    if model is None:
        model = DEFAULT_MODEL

    try:
        #添加用户问题到对话历史
        add_chart_history("user", prompt)

        responses=Generation.call(
            model=model,  # 模型名称
            messages=chat_history,  # 对话参数
            result_format="message",
            stream=True,  # 开启流式输出
            incremental_output=True  # 增量输出
        )

        # 完整回答内容
        full_answer = ""
        for response in responses:
            if response.status_code == 200:
                result = _extract_chunk_content(response)
                if result:
                    print(result, end="", flush=True)
                    # 拼接完整内容
                    full_answer += result
            else:
                print(f"错误: {response.status_code} - {response.message}")

        # 添加AI回复到对话历史
        if full_answer:
            add_chart_history("assistant", full_answer)
        else:
            print("未收到可用的模型回复内容")
    except Exception as e:
        print(f"请求失败：{str(e)}")

def main():
    """主函数"""
    #检查API KEY
    if not dashscope.api_key:
        print("错误：未找到API KEY")
        print("请确保：1. 存在.env文件 2. .env文件包含API_KEY=sk-xxx")
        return

    from ai_chat.rag_chat import RAGAssistant

    assistant=RAGAssistant()

    #知识库初始化
    if not assistant.init():
        return

    while True:
        #获取用户输入信息
        user_input=input("\n 用户：")

        if "/exit" == user_input:
            break
        #调用对话方法
        print("\n AI助手：")
        assistant.chat(user_input)

        print()

if __name__ == "__main__":
    main()

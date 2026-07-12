from ai_chat.llm_client import get_llm_model, stream_chat

#设置System_Message
SYSTEM_MESSAGE="你是一个公司的HR"

# 定义列表保存对话历史，并作为参数进行传递
chat_history = []

def add_chart_history(role,content):
    """添加对话到历史消息"""
    chat_history.append(
        {"role": role, "content": content}
    )


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
    try:
        #添加用户问题到对话历史
        add_chart_history("user", prompt)

        # 完整回答内容
        full_answer = ""
        for result in stream_chat(chat_history):
            print(result, end="", flush=True)
            full_answer += result

        # 添加AI回复到对话历史
        if full_answer:
            add_chart_history("assistant", full_answer)
        else:
            print("未收到可用的模型回复内容")
    except Exception as e:
        print(f"请求失败：{str(e)}")

def main():
    """主函数"""
    try:
        get_llm_model()
    except RuntimeError as error:
        print(f"错误：{error}")
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

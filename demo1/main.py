from ai_chat.chat import chat
from ai_chat.llm_client import get_llm_model


def main():
    """主函数"""
    try:
        model = get_llm_model()
    except RuntimeError as error:
        print(f"错误：{error}")
        print("请确保 .env 包含 LLM_API_KEY、LLM_BASE_URL 和 LLM_MODEL")
        return

    print(f"当前模型：{model}")

    while True:
        user_input = input("\n 用户：")

        if user_input == "/exit":
            break

        print("\n AI助手：")
        chat(user_input)
        print()


if __name__ == "__main__":
    main()

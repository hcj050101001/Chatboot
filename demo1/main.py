from ai_chat.chat import chat, dashscope


def main():
    """主函数"""
    if not dashscope.api_key:
        print("错误：未找到API KEY")
        print("请确保：1. 存在.env文件 2. .env文件包含API_KEY=sk-xxx")
        return

    while True:
        user_input = input("\n 用户：")

        if user_input == "/exit":
            break

        print("\n AI助手：")
        chat(user_input)
        print()


if __name__ == "__main__":
    main()

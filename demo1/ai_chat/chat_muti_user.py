"""
多用户会话助手——支持会话隔离
持久化存储对话内容
基于chat.py进行改造
"""
import json
import os
import uuid
from datetime import datetime

import dashscope
from dashscope import Generation
from dotenv import load_dotenv

#加载环境变量
load_dotenv()

#配置API_KEY
dashscope.api_key=os.getenv("API_KEY")
DEFAULT_MODEL=os.getenv("DEFAULT_MODEL")

#系统提示词 System_Message 规定模型额角色，功能列表，回答风格等
#用户提问的问题 User_Message
#模型的回答 Assistant_Message
#工具的回答 Tool_Message

#设置System_Message
SYSTEM_MESSAGE="你是一个AI助手"

#持久化存储用户对话记忆的文件名称路径
STORAGE_LIFE="session.json"

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

class Session:
    """单个用户的会话，每个Session对象作为一个对话用户"""

    def __init__(self,username):
        self.username= username
        self.session_id = str(uuid.uuid4())[:8] #截取转换为字符串的36位的前8位作为id
        self.created_at=datetime.now()
        self.last_active=datetime.now()
        self.chat_history=[
            {"role": "system", "content": SYSTEM_MESSAGE}
        ]

    def add_chat_history(self,role,content):
        """添加对话内容到对话历史"""
        self.chat_history.append({"role":role,"content":content})
        self.last_active=datetime.now()

    def get_history(self,max_turns=10):
        """获取历史消息，获取最大轮次数为max_turns的历史消息"""
        #保留System_message+最近的N轮对话
        system_msg=[self.chat_history[0]] if self.chat_history else []
        recent=self.chat_history[1:] #从除了系统提示词之后的对话历史记录
        recent=recent[-max_turns*2:] #然后从后往前获取最新的20条记录

        return system_msg+recent

    def clear_history(self):
        """清空对话历史"""
        self.chat_history=[
            {"role": "system", "content": SYSTEM_MESSAGE}
        ]

    def to_dict(self):
        """将对象转换成可以序列化保存的格式"""
        return{
            "username":self.username,
            "session_id":self.session_id,
            "create_at":self.created_at.isoformat(), #时间日期对象转换为字符串
            "last_active":self.last_active.isoformat(),
            "chat_history":self.chat_history
        }

class SessionManager:
    """会话管理器，管理所有用户会话"""

    def __init__(self):
        self.sessions={} #字典：保存用户名和对应的用户对象
        self.storage_file=STORAGE_LIFE

    def get_or_create(self,username):
        """创建或获取用户会话"""
        if username not in self.sessions:
            self.sessions[username]=Session(username)
            print(f"为新用户{username}创建会话")

        #用户之前存在
        self.sessions[username].last_active=datetime.now()

        return self.sessions[username]

    def save_to_file(self):
        """保存会话信息到文件（持久化）"""
        try:
            data={}

            for username,session in self.sessions.items():
                data[username]=session.to_dict()

            with open(self.storage_file,"w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2) #indent=2 格式化缩进

        except Exception as e:
            print(f"持久化保存会话失败:{e}")

class ChatMultiUser:

    def __init__(self):
        self.session_manager=SessionManager()

    def chat(self,username,prompt):
        """
        处理用户信息

        Args:
            username：用户名称
            prompt：用户问题
        """
        #获取用户会话对象
        session=self.session_manager.get_or_create(username)

        #添加用户问题到对话历史
        session.add_chat_history("user",prompt)

        #获取当前用户最近的对话历史作为和大模型交互参数
        history_msg=session.get_history(max_turns=5)

        try:

            responses = Generation.call(
                model=DEFAULT_MODEL,  # 模型名称
                messages=history_msg,  # 对话参数
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

            # 只有拿到有效内容时才写入历史
            if full_answer:
                session.add_chat_history("assistant", full_answer)
            else:
                print("未收到可用的模型回复内容")

            self.session_manager.save_to_file()
        except Exception as e:
            print(f"请求失败：{str(e)}")

def main():
    """主函数"""
    #检查API KEY
    if not dashscope.api_key:
        print("错误：未找到API KEY")
        print("请确保：1. 存在.env文件 2. .env文件包含API_KEY=sk-xxx")
        return
    if not DEFAULT_MODEL:
        print("错误：未找到DEFAULT_MODEL")
        print("请确保：.env文件包含DEFAULT_MODEL=你的模型名")
        return

    #获取多用户对话对象
    chat=ChatMultiUser()

    #当前登陆用户
    current_user=None

    print("="*50)
    print("多用户对话小助手")
    print("="*50)
    print()

    print("命令说明：")
    print("/login <用户名> -登陆/切换用户")
    print("/logout        -退出当前登陆")
    print("/exit          -退出程序")
    print()

    while True:
        #显示当前用户
        user_prompt=f"[{current_user or '未登录'}] 用户："

        try:
            user_input=input(user_prompt).strip() #去掉空格和换行
            if not user_input:
                 continue
            if user_input=="/exit":
                print("再见")
                break
            elif user_input.startswith("/login"):
                current_user=user_input[7:].strip()
                if current_user:
                    print(f"已登陆为：{current_user}")
                else:
                    print("用户名不能为空")
                    current_user=None
                continue
            elif user_input=="/logout":
                current_user=None
                print("已退出登陆")
                continue

            #未登录处理
            if not current_user:
                print(("请先使用 /login 进行登陆"))
                continue

            print("AI助手：")
            chat.chat(current_user,user_input)
            print()

        except EOFError:
            print("\n检测到输入结束，程序退出。")
            break
        except Exception as e:
            print(f"发生异常{str(e)}")

if __name__ == "__main__":
    main()

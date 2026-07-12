"""
多用户会话助手——支持会话隔离
持久化存储对话内容
基于chat.py进行改造
"""
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from ai_chat.llm_client import get_llm_model, stream_chat

#系统提示词 System_Message 规定模型额角色，功能列表，回答风格等
#用户提问的问题 User_Message
#模型的回答 Assistant_Message
#工具的回答 Tool_Message

#设置System_Message
SYSTEM_MESSAGE="""
        你是制造企业的智能小助手。
        负责公司制度与流程问答、采购订单、库存、生产、销售数据查询和已注册的图片检测等功能。
        数据问题必须依据工具结果，流程问题可以参考知识库中的内容。
        不要在回答中输出IMAGE_RESULT标记或图片路径，检测标注图由系统单独展示。
        输出必须使用清晰的纯文本结构，不使用Markdown星号、井号或表格。
        第一行给出“查询结论：”；每个字段单独一行；多条记录使用“【记录1】”分隔；最后单独一行写“数据来源：”。
"""

def strip_image_result_markers(content):
    """移除历史或模型回答中残留的内部图片标记。"""
    if not isinstance(content,str):
        return content

    content=re.sub(
        r"\[(?:IMAGE_RESULT|IAMGE_RESULT)\][\s\S]*?\[/(?:IMAGE_RESULT|IAMGE_RESULT)\]",
        "",
        content,
    )
    content=re.sub(r"\[/?(?:IMAGE_RESULT|IAMGE_RESULT)\]","",content)
    return content.rstrip()

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

class SessionManager:
    """会话管理器：缓存活跃会话，并将用户和消息持久化到本地 JSON。"""

    STORAGE_FILE=Path(__file__).resolve().parent / "chat_sessions.json"

    def __init__(self):
        self.sessions={}
        self._load_sessions()

    @staticmethod
    def _parse_datetime(value):
        try:
            return datetime.fromisoformat(value) if isinstance(value,str) else datetime.now()
        except ValueError:
            return datetime.now()

    def _load_sessions(self):
        """从本地 JSON 恢复所有会话；缺失或损坏文件不影响服务启动。"""
        if not self.STORAGE_FILE.exists():
            return
        try:
            with self.STORAGE_FILE.open("r",encoding="utf-8") as file:
                data=json.load(file)
            if not isinstance(data,dict):
                raise ValueError("会话文件根节点必须是对象")
            for username,record in data.items():
                if not isinstance(record,dict) or not isinstance(username,str):
                    continue
                session=Session(username)
                session.session_id=str(record.get("session_id") or session.session_id)
                session.created_at=self._parse_datetime(record.get("created_at"))
                session.last_active=self._parse_datetime(record.get("last_active"))
                history=record.get("chat_history",[])
                session.chat_history=[{"role":"system","content":SYSTEM_MESSAGE}]
                if isinstance(history,list):
                    for message in history:
                        if not isinstance(message,dict) or message.get("role") not in {"user","assistant"}:
                            continue
                        content=str(message.get("content",""))
                        if message["role"] == "assistant":
                            content=strip_image_result_markers(content)
                        session.chat_history.append({"role":message["role"],"content":content})
                self.sessions[username]=session
        except (OSError,json.JSONDecodeError,ValueError) as error:
            print(f"读取 JSON 会话记录失败，将使用空会话：{error}")

    def _save_sessions(self):
        """原子写入会话文件，避免写入中断导致历史文件损坏。"""
        data={
            username:{
                "username":session.username,
                "session_id":session.session_id,
                "created_at":session.created_at.isoformat(),
                "last_active":session.last_active.isoformat(),
                "chat_history":[
                    {"role":message["role"],"content":strip_image_result_markers(message["content"])}
                    for message in session.chat_history
                    if message.get("role") != "system"
                ],
            }
            for username,session in self.sessions.items()
        }
        temporary_file=self.STORAGE_FILE.with_suffix(".tmp")
        try:
            with temporary_file.open("w",encoding="utf-8") as file:
                json.dump(data,file,ensure_ascii=False,indent=2)
            temporary_file.replace(self.STORAGE_FILE)
            return True
        except OSError as error:
            print(f"JSON 会话持久化失败：{error}")
            return False

    def get_or_create(self,username):
        """创建或获取用户会话"""
        username=username.strip()
        if not username:
            raise ValueError("用户名不能为空")
        if len(username) > 100:
            raise ValueError("用户名不能超过 100 个字符")

        if username not in self.sessions:
            session=Session(username)
            print(f"为新用户{username}创建 JSON 会话")
            self.sessions[username]=session

        session=self.sessions[username]
        session.last_active=datetime.now()
        return session

    def save_session(self,username):
        """将当前所有会话保存到本地 JSON 文件。"""
        return username in self.sessions and self._save_sessions()

    def clear_history(self,username):
        """清空指定用户的对话历史"""
        try:
            session=self.sessions.get(username)
            if session is None:
                return False
            session.clear_history()
            session.last_active=datetime.now()
            self.sessions[username]=session
            return self.save_session(username)
        except Exception as error:
            print(f"清空 JSON 会话历史失败：{error}")
            return False

    def list_users(self):
        """统计所有用户信息"""
        users=sorted(self.sessions,key=lambda username:self.sessions[username].last_active,reverse=True)
        return {"total_users":len(users),"users":users}

class ChatMultiUser:

    def __init__(self):
        self.session_manager=SessionManager()

    def show_user(self):
        """显示统计信息"""
        users_info=self.session_manager.list_users()
        print(f"\n统计信息：")
        print(f"总用户数：{users_info['total_users']}")
        if users_info['users']:
            print(f"用户列表：{','.join(users_info['users'])}")

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

            # 完整回答内容
            full_answer = ""
            for result in stream_chat(history_msg):
                print(result, end="", flush=True)
                full_answer += result

            # 只有拿到有效内容时才写入历史
            if full_answer:
                session.add_chat_history("assistant", full_answer)
            else:
                print("未收到可用的模型回复内容")

            self.session_manager.save_session(username)
        except Exception as e:
            print(f"请求失败：{str(e)}")

def main():
    """主函数"""
    try:
        get_llm_model()
    except RuntimeError as error:
        print(f"错误：{error}")
        return

    #获取多用户对话对象
    chat=ChatMultiUser()

    #当前登录用户
    current_user=None

    print("="*50)
    print("多用户对话小助手")
    print("="*50)
    print()

    print("命令说明：")
    print("/login <用户名> -登录/切换用户")
    print("/logout        -退出当前登录")
    print("/clear         -清空当前用户的对话历史")
    print("/users         -显示所有用户的统计信息")
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
                    print(f"已登录为：{current_user}")
                else:
                    print("用户名不能为空")
                    current_user=None
                continue
            elif user_input=="/logout":
                current_user=None
                print("已退出登录")
                continue
            elif user_input=="/users":
                chat.show_user()
                continue
            elif user_input=="/clear":
                if current_user:
                    if chat.session_manager.clear_history(current_user):
                        print(f"已清空{current_user}的对话历史")
                    else:
                        print(f"用户{current_user}没有对话历史")
                else:
                    print("请先使用 /login 进行登录")
                continue

            #未登录处理
            if not current_user:
                print(("请先使用 /login 进行登录"))
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

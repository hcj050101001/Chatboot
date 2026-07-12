"""
多用户会话助手——支持会话隔离
持久化存储对话内容
基于chat.py进行改造
"""
import os
import re
import uuid
from datetime import datetime

import pymysql
from dotenv import load_dotenv

from ai_chat.llm_client import get_llm_model, stream_chat

#加载环境变量
load_dotenv()

#系统提示词 System_Message 规定模型额角色，功能列表，回答风格等
#用户提问的问题 User_Message
#模型的回答 Assistant_Message
#工具的回答 Tool_Message

#设置System_Message
SYSTEM_MESSAGE="""
        你是制造企业的智能ERP助手。
        主要回答采购订单、库存、生产工单、销售合同和ERP操作流程问题。
        数据问题必须依据工具结果，流程问题可以参考知识库中的内容。
        不要在回答中输出IMAGE_RESULT标记或图片路径，检测标注图由系统单独展示。
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
    """会话管理器：缓存活跃会话，并将用户和消息持久化到 MySQL。"""

    def __init__(self):
        self.sessions={}
        self._ensure_tables()

    @staticmethod
    def _get_connection():
        """按需创建 ERP 数据库连接。"""
        return pymysql.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            charset="utf8mb4",
        )

    def _ensure_tables(self):
        """创建 MySQL 会话表和消息表；已有表不会被修改或删除。"""
        conn=None
        try:
            conn=self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        username VARCHAR(100) NOT NULL,
                        session_id VARCHAR(36) NOT NULL,
                        created_at DATETIME NOT NULL,
                        last_active DATETIME NOT NULL,
                        PRIMARY KEY (username),
                        KEY idx_chat_sessions_last_active (last_active)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ERP聊天会话'
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id BIGINT NOT NULL AUTO_INCREMENT,
                        username VARCHAR(100) NOT NULL,
                        role VARCHAR(20) NOT NULL,
                        content LONGTEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        PRIMARY KEY (id),
                        KEY idx_chat_messages_username_id (username, id),
                        CONSTRAINT fk_chat_messages_session
                            FOREIGN KEY (username) REFERENCES chat_sessions(username)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ERP聊天消息'
                """)
            conn.commit()
        except Exception as error:
            raise RuntimeError(f"初始化 MySQL 会话存储失败：{error}") from error
        finally:
            if conn:
                conn.close()

    def _load_session_from_database(self,username):
        """从 MySQL 恢复一个用户的会话和历史消息。"""
        conn=None
        try:
            conn=self._get_connection()
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(
                    """SELECT username, session_id, created_at, last_active
                       FROM chat_sessions WHERE username=%s""",
                    (username,),
                )
                session_row=cursor.fetchone()
                if not session_row:
                    return None

                cursor.execute(
                    """SELECT role, content FROM chat_messages
                       WHERE username=%s ORDER BY id""",
                    (username,),
                )
                message_rows=cursor.fetchall()

            session=Session(username)
            session.session_id=session_row["session_id"]
            session.created_at=session_row["created_at"]
            session.last_active=session_row["last_active"]
            session.chat_history=[{"role":"system","content":SYSTEM_MESSAGE}]
            for message in message_rows:
                content=message["content"]
                if message["role"] == "assistant":
                    content=strip_image_result_markers(content)
                session.chat_history.append({"role":message["role"],"content":content})
            return session
        finally:
            if conn:
                conn.close()

    def _save_session_to_database(self,session):
        """以一个事务写入会话元数据和完整消息历史。"""
        conn=None
        try:
            conn=self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_sessions (username, session_id, created_at, last_active)
                    VALUES (%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        session_id=VALUES(session_id),
                        last_active=VALUES(last_active)
                    """,
                    (session.username,session.session_id,session.created_at,session.last_active),
                )
                cursor.execute("DELETE FROM chat_messages WHERE username=%s",(session.username,))
                messages=[
                    (session.username,message["role"],strip_image_result_markers(message["content"]),datetime.now())
                    for message in session.chat_history
                    if message.get("role") != "system"
                ]
                if messages:
                    cursor.executemany(
                        """INSERT INTO chat_messages (username, role, content, created_at)
                           VALUES (%s,%s,%s,%s)""",
                        messages,
                    )
            conn.commit()
            return True
        except Exception as error:
            if conn:
                conn.rollback()
            print(f"MySQL 会话持久化失败：{error}")
            return False
        finally:
            if conn:
                conn.close()

    def get_or_create(self,username):
        """创建或获取用户会话"""
        username=username.strip()
        if not username:
            raise ValueError("用户名不能为空")
        if len(username) > 100:
            raise ValueError("用户名不能超过 100 个字符")

        if username not in self.sessions:
            session=self._load_session_from_database(username)
            if session is None:
                session=Session(username)
                print(f"为新用户{username}创建 MySQL 会话")
            else:
                print(f"已从 MySQL 恢复用户{username}的会话")
            self.sessions[username]=session

        session=self.sessions[username]
        session.last_active=datetime.now()
        return session

    def save_session(self,username):
        """保存指定用户会话到 MySQL。"""
        session=self.sessions.get(username)
        return self._save_session_to_database(session) if session else False

    def clear_history(self,username):
        """清空指定用户的对话历史"""
        try:
            session=self.sessions.get(username) or self._load_session_from_database(username)
            if session is None:
                return False
            session.clear_history()
            session.last_active=datetime.now()
            self.sessions[username]=session
            return self.save_session(username)
        except Exception as error:
            print(f"清空 MySQL 会话历史失败：{error}")
            return False

    def list_users(self):
        """统计所有用户信息"""
        conn=None
        try:
            conn=self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT username FROM chat_sessions ORDER BY last_active DESC")
                users=[row[0] for row in cursor.fetchall()]
            return {"total_users":len(users),"users":users}
        finally:
            if conn:
                conn.close()

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

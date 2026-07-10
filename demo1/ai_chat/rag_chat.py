"""
RAG知识库问答系统 - 支持docs 文件夹多文档检索
注意事项：
    1.确保.env文件已经配置
    2.创建docs文件夹，讲PDF/WORD/TXT文件放入
    3.首次运行会自动构建知识库
"""
import os
from pathlib import Path

import dashscope
from dashscope import Generation
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_chat.chat import _extract_chunk_content

#加载环境变量
load_dotenv()

dashscope.api_key=os.getenv("API_KEY")
DEFAULT_MODEL=os.getenv("DEFAULT_MODEL") #对话模型
EL_MODEL=os.getenv("EL_MODEL") #向量模型

#路径配置：始终以当前脚本所在目录为准，避免受启动目录影响
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = str(BASE_DIR / "docs")
VECTOR_DB_PATH = str(BASE_DIR / "chroma_db")

#支持的文档加载器
LOADER_MAPPING={
    ".pdf":PyPDFLoader,
    ".txt":TextLoader,
    ".docx":Docx2txtLoader
}

#设置System_Message
SYSTEM_MESSAGE="你是公司的人事小助手，专门回答请假，入职，离职，考勤，薪资等人事问题，不会的问题不要回答"

class RAGAssistant:
    """基于知识库的问答助手 - 支持多文档"""
    def __init__(self,docs_dir=DOCS_DIR,db_path=VECTOR_DB_PATH):
        self.docs_dir=docs_dir
        self.file_list=[] #记录已经加载的文档
        self.db_path=db_path #向量存放路径
        self.vectorstore=None #用于后续存储向量数据库对象

    def get_all_documents(self):

        """加载docs文件夹下所有支持的文档"""
        self.file_list=[]

        #接受所有文档块的列表
        all_documents=[]

        #检查目录文档是否存在，不存在进行目录创建
        if not os.path.exists((self.docs_dir)):
            os.makedirs(self.docs_dir, exist_ok=True)
            print(f"已创建{self.docs_dir}文件夹，请将文档放入之后重新运行")
            return

        #遍历加载文件夹中的所有支持文件
        for filename in os.listdir(self.docs_dir):
            #文档完整路径
            file_path=os.path.join(self.docs_dir,filename)
            ext=os.path.splitext(filename)[1].lower() #文档后缀

            #判断是否支持加载该文档
            if ext in LOADER_MAPPING:
                print(f"加载文档：{filename}")

                try:
                    loader_class=LOADER_MAPPING[ext] #文件加载器类型

                    #不同文档构建不同的加载器对象
                    if ext==".txt":
                        loader=loader_class(file_path,encoding="utf-8")
                    else:
                        loader=loader_class(file_path)

                    #加载文档返回文档块列表
                    documents=loader.load()

                    # print(f"加载的文档快的第一个内容：{documents[0]}")

                    #为每个文档块的source来源重新赋值，去掉路径保留名称
                    for doc in documents:
                        doc.metadata["source"]=filename

                    #将加载的文档放到总列表中
                    all_documents.extend(documents)
                    self.file_list.append(filename) #记录加载的文档名称

                    print((f"{filename}共{len(documents)}页/段"))

                except Exception as e:
                    print(f"加载失败{e}")

            else:
                    print(f"跳过不支持加载的文件：{filename}(支持 .pdf, .docx , .txt)")

        return all_documents

    def build_knowledge_base(self):
        """构建向量知识库"""

        #所有文档块
        documents=self.get_all_documents()

        if not documents:
            print("\n 未找到任何支持的文档")
            print(f"请将PDF/WORD/TXT文件放入{self.docs_dir}文件夹后重新运行")
            return False

        print(f"共加载{len(documents)}个文档片段")

        #将文档片段分割成文档块
        print("正在进行文档分块...")
        #构建切割对象
        txt_splitter=RecursiveCharacterTextSplitter(
            chunk_size=500, #每块多少字符
            chunk_overlap=100, #上下块重叠的字符
            separators=["\n\n","\n","。",";",","," ",""] #优先分割的符号
        )

        #文档块分割成文档片段
        chunks=txt_splitter.split_documents(documents)
        print(f"共分为{len(chunks)}个文档块")

        #构建向量化对象
        print("正在进行向量化...")
        embeddings=DashScopeEmbeddings(
            model=EL_MODEL,
            dashscope_api_key=dashscope.api_key
        )

        print("正在存入向量数据库...")

        #构建向量数据库对象
        self.vectorstore=Chroma.from_documents(
            documents=chunks, #文档块
            embedding=embeddings, #向量模型
            persist_directory=self.db_path #持久化存储路径
        )

        print("\n 知识库构建完成！")
        print(f"文档目录：{self.docs_dir}")
        print(f"已加载文件：{self.file_list}")
        print(f"总文本块数:{len(chunks)}")
        return True

    def load_knowledge_base(self):
        """加载已经构建的知识库"""

        # 构建向量化对象
        print("正在进行向量化...")
        embeddings = DashScopeEmbeddings(
            model=EL_MODEL,
            dashscope_api_key=dashscope.api_key
        )

        #从持久化路径中读取向量数据
        self.vectorstore=Chroma(
            persist_directory=self.db_path, #从指定的持久化路径构建向量数据库
            embedding_function=embeddings #传入向量检索函数需要的向量对象
        )

        print("知识库加载成功...")

        return True

    def init(self):
        """知识库初始化，是构建还是加载"""
        if not os.path.exists(self.db_path) or not os.listdir(self.db_path):
            return self.build_knowledge_base()
        else:
            return self.load_knowledge_base()

    def search_documents(self,question,k=3):
        """
        根据用户问题从知识库检索相似片段
        Args：
            question:用户问题
            k：最多检索出几个相似片段
        Returns：
            检索出的相似片段列表
        """

        docs=self.vectorstore.similarity_search(question,k=k)

        return docs

    def chat(self,question):
        """基于知识库检索回答问题"""

        #通过问题检索文档
        docs=self.search_documents(question)

        #将检索到的文档片段添加来源并拼接为字符串
        context_parts=[]

        #遍历文档片段enumerate：指定遍历目标并指定索引初始编号
        for i,doc in enumerate(docs,1):
            source=doc.metadata.get("source","未知来源")
            context_parts.append(f"【文档片段{i} - 来自 《{source}》】 \n {doc.page_content}")

        context="\n\n".join(context_parts)

        #构建增强提示词prompt
        user_message=f"""
                【公司制度文档】
                {context}
                
                【员工问题】
                {question}
                
                请根据上述文档内容回答，并尽量标注信息来源文件
        """

        try:

            #将系统提示词和用户问题封装为模型参数
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message}
            ]

            responses = Generation.call(
                model=DEFAULT_MODEL,  # 模型名称
                messages=messages,  # 对话参数
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

            if full_answer:
                return full_answer
            else:
                print("未收到可用的模型回复内容")
        except Exception as e:
            print(f"请求失败：{str(e)}")

def main():
    """启动知识库问答命令行。"""
    if not dashscope.api_key:
        print("错误：未找到API KEY")
        print("请确保 .env 文件包含 API_KEY=sk-xxx")
        return

    if not DEFAULT_MODEL or not EL_MODEL:
        print("错误：未找到 DEFAULT_MODEL 或 EL_MODEL 配置")
        return

    rag = RAGAssistant()
    if not rag.init():
        return

    while True:
        try:
            question = input("\n用户：").strip()
        except EOFError:
            break

        if question == "/exit":
            break
        if not question:
            continue

        print("\nAI助手：")
        rag.chat(question)
        print()

if __name__ == "__main__":
    main()

# 智能小助手项目

本项目包含公司制度知识库问答、ERP 采购/库存/销售/生产数据查询、多用户 JSON 会话记录和零件图片检测功能。

## 推荐运行环境

- Windows 10/11
- Python 3.10、3.11 或 3.12（已在 Python 3.12 验证）
- MySQL 8.0（仅 ERP 数据查询需要）
- 可访问所配置的大模型、DashScope 和图片检测接口的网络

不要复制提交本机的 `.venv` 文件夹。虚拟环境包含本机绝对路径，换电脑后不能复用。

## 最简单的启动方式

1. 双击 `start.bat`。
2. 首次运行会自动创建 `.venv` 并安装依赖。
3. 如果项目中没有 `.env`，脚本会根据 `.env.example` 生成一份并打开记事本。
4. 填写配置后，再次双击 `start.bat`。
5. 浏览器会自动打开 `http://127.0.0.1:8001/`。

也可以在 PyCharm 中把项目解释器设为 `.venv/Scripts/python.exe`，然后直接运行 `main.py`。

## 必填配置

复制 `.env.example` 为 `.env` 后填写：

- `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`：对话模型配置。
- `DASHSCOPE_API_KEY`、`EL_MODEL`：知识库文本向量模型配置。
- `DB_HOST`、`DB_PORT`、`DB_USER`、`DB_PASSWORD`、`DB_NAME`：MySQL 配置。
- `DETECT_API_URL`：零件图片检测服务地址；默认使用项目演示接口。
- `OPEN_BROWSER`：默认为 `1`，设为 `0` 时启动服务但不自动打开浏览器。

`.env` 含密钥和密码，不应上传到公开仓库。提交给老师时可以让老师填写自己的密钥；若必须现场演示，请通过私密方式提供临时密钥并在演示后撤销。

## 首次启动会做什么

- `setup_database.py` 自动创建 `erp_db`、8 张 ERP 表和演示数据，不再使用 MySQL 保存聊天记录。
- 聊天记录自动保存到 `ai_chat/chat_sessions.json`，不同用户名相互隔离。
- 如果 `ai_chat/chroma_db_v4` 不存在或集合为空，会从 `ai_chat/docs` 中的文档自动构建知识库。

数据库账号需要具有 `CREATE DATABASE`、建表和读写演示数据的权限。如果数据库初始化失败，网页和知识库问答仍能启动，但 ERP 数据查询会提示数据库不可用。

## 测试问题

- `/login 老师`
- `试用期是多久？`
- `A供应商有哪些采购订单还未到货？`
- `当前库存低于安全库存的商品有哪些？`
- `2026-07 销售额最高的客户是谁？`
- `查看 products 表结构`

## 手动命令

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_database.py
.\.venv\Scripts\python.exe main.py
```

停止服务：在运行窗口按 `Ctrl+C`。

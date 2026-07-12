"""项目启动入口。

- 运行方式：python main.py
- 或使用 uvicorn：uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import logging

import uvicorn

from app.config import get_settings

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

if __name__ == '__main__':
    settings = get_settings()

    print()
    print('=' * 60)
    print('  小杉机器人 LangChain 智能体')
    print('=' * 60)
    print(f'  模型: {settings.dashscope_model}')
    print(f'  监听: http://{settings.host}:{settings.port}')
    print(f'  接口: http://{settings.host}:{settings.port}/v1/chat/completions')
    print('=' * 60)
    print()
    print('  等待小杉机器人连接...')
    print()

    uvicorn.run(
        'app.server:app',
        host=settings.host,
        port=settings.port,
        reload=True,
    )

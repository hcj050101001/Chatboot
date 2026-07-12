"""
配置管理模块，读取.env文件中的配置并设置默认值
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

#继承BaseSettings类:
# 1.读取环境变量自动去os.env中或者.env找同名变量
#2.将配置文件中的字符串值自动转为需要的声明的配置

class Settings():
    """项目运行时配置"""

    #Filed:字段描述器对象  ...从.env中取值，alias:对应的属性名
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL")
    dashscope_model = os.getenv("DASHSCOPE_MODEL")

    #服务配置
    host = os.getenv("HOST")
    port = int(os.getenv("PORT"))


#改成单例模式，缓存最近一次的调用结果
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的配置对象"""
    return Settings()
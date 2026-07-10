"""
工具模块 - 用于封装挂载到大模型的各种工具
"""
from datetime import datetime

from langchain_core.tools import tool

@tool
def get_current_datetime(format_type:str="full") ->str:
    """
    获取当前的日期和时间。
    当用户询问“今天几号”，“现在几点”，“当前时间”，“今天是星期几”等问题的时候调用
    Args:
        format_type:返回的时间格式，可选值：
        - "full"：完整的日期时间(默认值) 比如 "2026-07-10 16:49:00 星期五"
        - "date"：仅日期 比如 "2026-07-10"
        - "time"：仅时间 比如 "16:49:00"
        - "datetime"：日期和时间 比如 "2026-07-10 16:49:00"
        - "weekday"：仅星期 比如 "星期五"
    Returns:
        格式化之后的日期时间字符串
    """

    now=datetime.now()

    #星期映射
    weekdays=["星期一","星期二","星期三","星期四","星期五","星期六","星期日",]
    weekday=weekdays[now.weekday()]

    format_map={
        "full":f"{now.strftime('%Y-%m-%d %H:%M:%S')} {weekday}",
        "date": f"{now.strftime('%Y-%m-%d')}",
        "time": f"{now.strftime('%H:%M:%S')}",
        "datetime": f"{now.strftime('%Y-%m-%d %H:%M:%S')}",
        "weekday": f"{weekday}",
    }

    return format_map.get(format_type)

#工具列表
TOOLS=[
    get_current_datetime
]

def excute_tool(tool_name:str,**kwargs):
    """
    根据工具名称执行对应的工具函数
    Args:
        tool_name：工具名称
        **kwargs：工具参数
    Returns:
        工具的执行结果(字符串)
    """
    for tool in TOOLS:
        if tool.name==tool_name:
            return tool.invoke(kwargs)

    return f"未找到可执行的工具{tool_name}"

if __name__ == "__main__":
    result=get_current_datetime.invoke({"format_type": "full"})
    print(result)
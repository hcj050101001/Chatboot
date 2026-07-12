"""
工具模块 - 用于封装挂载到大模型的各种工具
"""
import base64
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import dashscope
import httpx
from dashscope import Generation
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import ValidationError

#加载环境变量
load_dotenv()
dashscope.api_key=os.getenv("API_KEY")
DEFAULT_MODEL=os.getenv("DEFAULT_MODEL")

#标注图保存目录：与 FastAPI 挂载的 ai_chat/static 保持一致
BASE_DIR = Path(__file__).resolve().parent
ANNOTATED_IMAGE_DIR = BASE_DIR / "static" / "annotated"

#确保目录存在
os.makedirs(ANNOTATED_IMAGE_DIR,exist_ok=True)

#智能检测地址
DETECT_API_URL="http://47.104.167.34:9900/api/detect"


@tool
def get_current_datetime(
    format_type: Literal["full", "date", "time", "datetime", "weekday"] = "full",
) -> str:
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

    result = format_map.get(format_type)
    if result is None:
        supported_formats = ", ".join(format_map)
        return f"不支持的时间格式：{format_type}。可选值：{supported_formats}"

    return result

def save_annotated_image(image_base64:str)->str:
    """
    将base标注图保存为本地文件，返回访问的URL
    """
    if not image_base64:
        return""

    try:
        #解码base64
        image_bytes=base64.b64decode(image_base64)

        #生成文件名称 时间戳 + 随机字符串
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id=str(uuid.uuid4())[:8]
        filename=f"annotated_{timestamp}_{unique_id}.jpg"
        filepath=os.path.join(ANNOTATED_IMAGE_DIR,filename)

        #保存文件
        with open(filepath,"wb") as f:
            f.write(image_bytes)

        #返回可以访问的url
        return f"/static/annotated/{filename}"

    except Exception as e:
        print(f"标注图保存失败:{e}")
        return ""

@tool
def detect_defect(image_data:str)->str:
    """
    检测零件图片中的瑕疵。当用户上传零件图片并询问"有没有问题"、"检测一下"时使用

    Args：
        image_data:图片的字节数据(从请求中获取的file内容)

    Returns：
        检测结果摘要
    """
    try:

        image_bytes=base64.b64decode(image_data)

        #构造参数
        params={
            "file":("image.jpg",image_bytes,"image/jpeg")
        }

        #发送请求
        with httpx.Client(timeout=10.0) as client:
            response=client.post(DETECT_API_URL,files=params)
            #检测响应是否成功
            response.raise_for_status()
            data=response.json()

            #提取标注图
            annotated_image=data.get("annotated_image","")

            image_url=""
            if annotated_image:
                image_url=save_annotated_image(annotated_image)
                print(f"标注图已保存:{image_url}")

            boxes=data.get("boxes",[]) #缺陷详情
            num=int(data.get("num_detections",0) or 0) #缺陷数量

            if num==0:
                result_text="零件表面无瑕疵，检测合格"
            else:
                line=[f"检测到{num}处瑕疵:"]

                for i,b in enumerate(boxes,1):
                    name=b.get("class_name_cn") #缺陷名称中文
                    conf=b.get("confidence",0) * 100 #检测可信度百分比
                    line.append(f"    {i}.{name}(置信度 {conf:.1f}%)")

                #添加建议
                high_conf=[b for b in boxes if b.get("confidence",0)>0.5]

                if high_conf:
                    line.append("\n建议：存在高置信度缺陷，建议重点检查或返工。")
                else:
                    line.append("\n建议：缺陷置信度较低，建议人工复核确认。")

                result_text="\n".join(line)

            #返回json格式内容字符串，包含文本和检测图片
            return json.dumps({
                "result":result_text,
                "image_url":image_url
            },ensure_ascii=False)

    except Exception as e:
        print(f"调用图片检测工具异常：{e}")
        return json.dumps({
            "result":f"检测失败:{str(e)}",
            "image_url":""
        },ensure_ascii=False)

#工具列表
TOOLS=[
    get_current_datetime,
    detect_defect
]

def _find_tool(tool_name: str):
    """根据名称查找已注册的工具。"""
    return next((tool for tool in TOOLS if tool.name == tool_name), None)


def _validate_tool_params(tool, params: dict) -> dict:
    """校验工具参数，拒绝模型返回的未知参数。"""
    if not isinstance(params, dict):
        raise ValueError("params 必须是 JSON 对象")

    allowed_params = set(tool.args_schema.model_fields)
    unknown_params = set(params) - allowed_params
    if unknown_params:
        raise ValueError(f"不支持的参数：{', '.join(sorted(unknown_params))}")

    validated_params = tool.args_schema.model_validate(params)
    return validated_params.model_dump()


def _parse_tool_decision(content: str) -> dict:
    """从模型返回中提取并解析 JSON 决策结果。"""
    if not isinstance(content, str):
        raise ValueError("模型返回内容不是字符串")

    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1]).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or start > end:
        raise ValueError("模型返回中没有 JSON 对象")

    result = json.loads(content[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("工具决策必须是 JSON 对象")

    return result


def excute_tool(tool_name:str,**kwargs):
    """
    根据工具名称执行对应的工具函数
    Args:
        tool_name：工具名称
        **kwargs：工具参数
    Returns:
        工具的执行结果(字符串)
    """
    tool = _find_tool(tool_name)
    if tool is None:
        return f"未找到可执行的工具{tool_name}"

    try:
        params = _validate_tool_params(tool, kwargs)
    except (ValidationError, ValueError, TypeError) as error:
        return f"工具参数无效：{error}"

    return tool.invoke(params)

def decide_tool(question:str)->tuple:
    """
    让AI决定是否调用工具
    返回：(tool_name,params_dict)或(None,None)
    """

    #构建工具描述列表
    tool_list=[]
    for tool in TOOLS:
        tool_list.append({
            "name":tool.name,
            "description":tool.description,
            "parameters":tool.args
        })

    prompt=f"""
        你是一个智能助手，根据用户问题来决定是否调用工具
        可用工具：
        {json.dumps(tool_list,ensure_ascii=False,indent=2)}

        用户问题：{question}

        请严格以纯JSON的格式输出决策结果，不要包含任何其他描述文字：
        {{"tool":"工具名称","params":{{"参数名":"参数值"}}}}

        如果不需要调用任何工具，输出：
        {{"tool":"none","params":{{}}}}

        规则：
        1.只选择一个最相关的工具
        2.参数从用户问题中提取
        3.如果用户问的是制度问题(如请假，考勤，入离职)，不要调用工具
        4.只输出纯JSON，不要有任何解释
    """

    response=Generation.call(
        model=DEFAULT_MODEL,
        messages=[{"role":"user","content":prompt}],
        result_format="message"
    )

    status_code = getattr(response, "status_code", None)
    if status_code != 200:
        message = getattr(response, "message", "未知错误")
        print(f"AI工具决策请求失败：{status_code} - {message}")
        return None,None

    try:
        content=response.output.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        print(f"AI工具决策响应格式异常：{error}")
        return None,None

    print(f"AI工具使用原始返回：{content[:200]}....")

    try:
        result = _parse_tool_decision(content)
    except (json.JSONDecodeError, ValueError) as error:
        print(f"AI工具决策 JSON 解析失败：{error}")
        return None,None

    tool_name = result.get("tool", "none")
    params = result.get("params", {})
    if tool_name == "none":
        return None,None

    if not isinstance(tool_name, str):
        print("AI工具决策无效：tool 必须是字符串")
        return None,None

    tool = _find_tool(tool_name)
    if tool is None:
        print(f"AI工具决策无效：未注册的工具 {tool_name}")
        return None,None

    try:
        params = _validate_tool_params(tool, params)
    except (ValidationError, ValueError, TypeError) as error:
        print(f"AI工具决策参数无效：{error}")
        return None,None

    return tool_name,params

if __name__ == "__main__":
    imgbs64="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"
    print(save_annotated_image(imgbs64))

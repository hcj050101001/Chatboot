"""基于 OpenAI Chat Completions 协议的统一大模型客户端。"""
import json
import os
from collections.abc import Iterator

import httpx
from dotenv import load_dotenv

load_dotenv()


def _config() -> tuple[str, str, str]:
    """读取 URL、密钥和模型名；兼容原有 API_KEY 配置。"""
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
    model = os.getenv("LLM_MODEL") or os.getenv("DEFAULT_MODEL")
    if not api_key:
        raise RuntimeError("未配置 LLM_API_KEY 或 API_KEY")
    if not model:
        raise RuntimeError("未配置 LLM_MODEL 或 DEFAULT_MODEL")
    return base_url, api_key, model


def get_llm_model() -> str:
    """返回当前启用的模型名称。"""
    return _config()[2]


def complete_chat(messages: list[dict], json_mode: bool = False) -> str:
    """执行一次非流式 Chat Completions 请求并返回助手文本。"""
    base_url, api_key, model = _config()
    payload = {"model": model, "messages": messages, "stream": False}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
        response = client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    try:
        return data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"模型响应格式异常：{json.dumps(data, ensure_ascii=False)[:500]}") from error


def stream_chat(messages: list[dict]) -> Iterator[str]:
    """执行流式 Chat Completions 请求，逐段产出可展示文本。"""
    base_url, api_key, model = _config()
    payload = {"model": model, "messages": messages, "stream": True}

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0].get("delta", {}).get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                    raise RuntimeError(f"流式模型响应格式异常：{data[:500]}") from error
                if content:
                    yield content

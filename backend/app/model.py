import os
import re
import json
import contextvars
from typing import AsyncGenerator, Optional

import httpx


LLAMA_HOST = os.getenv("LLAMA_HOST", "http://127.0.0.1:8080")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
N_PREDICT = int(os.getenv("N_PREDICT", "2048"))
TIMEOUT = 120.0

_tool_calls_var: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar('tool_calls', default=None)


def extract_tool_calls(text: str) -> list:
    pattern = r'<\|tool_call\|>\s*(\{.*?\})\s*<\|end_tool_call\|>'
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for i, m in enumerate(matches):
        try:
            data = json.loads(m)
            calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": data["name"], "arguments": json.dumps(data.get("arguments", {}))}
            })
        except Exception:
            continue
    return calls


def get_last_tool_calls() -> Optional[list]:
    calls = _tool_calls_var.get()
    _tool_calls_var.set(None)
    return calls


async def generate(messages: list[dict], max_tokens: int = N_PREDICT, temperature: float = TEMPERATURE, stream: bool = True, tools: list = None) -> AsyncGenerator[str, None]:
    _tool_calls_var.set(None)

    url = f"{LLAMA_HOST}/v1/chat/completions"
    payload = {"model": "llama", "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": stream}
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            choice = chunk.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                _tool_calls_var.set(tool_calls)
                            if content:
                                yield content
                        except Exception:
                            continue
        except httpx.ReadTimeout:
            yield "\n\n[Error: Request timed out.]"


async def generate_with_image(text: str, image_b64: str, media_type: str = "image/jpeg", max_tokens: int = N_PREDICT, temperature: float = TEMPERATURE) -> AsyncGenerator[str, None]:
    url = f"{LLAMA_HOST}/v1/chat/completions"
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}}, {"type": "text", "text": text}]}]
    payload = {"model": "llama", "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except httpx.ReadTimeout:
            yield "\n\n[Error: Request timed out.]"

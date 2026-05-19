import os
from typing import AsyncGenerator
import httpx


LLAMA_HOST = os.getenv("LLAMA_HOST", "http://127.0.0.1:8080")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
N_PREDICT = int(os.getenv("N_PREDICT", "2048"))
TIMEOUT = 120.0


async def generate(messages: list[dict], max_tokens: int = N_PREDICT, temperature: float = TEMPERATURE, stream: bool = True) -> AsyncGenerator[str, None]:
    url = f"{LLAMA_HOST}/v1/chat/completions"
    payload = {"model": "llama", "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": stream}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    import json
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except:
                        continue


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
                        import json
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except:
                            continue
        except httpx.ReadTimeout:
            yield "\n\n[Error: Request timed out.]"
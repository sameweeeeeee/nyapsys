import os
import json
from typing import AsyncGenerator, Optional

from app import db, rag, model, ingest
from app.tools import TOOLS, call_tool


MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))
SYSTEM_MESSAGE = "You are Nyapsys, a self-hosted AI assistant running locally on your Mac. You answer questions accurately, read and analyse files, and understand images. Be concise but thorough. If you are unsure, say so."


async def stream_from_model(messages: list[dict], max_tokens: int = 2048, temperature: float = 0.7) -> AsyncGenerator[str, None]:
    async for token in model.generate(messages=messages, max_tokens=max_tokens, temperature=temperature):
        yield token


async def run(user_message: str, conversation_id: str, file_bytes: Optional[bytes] = None,
              filename: Optional[str] = None, file_type: Optional[str] = None) -> AsyncGenerator[str, None]:
    conv = await db.get_messages(conversation_id)
    if not conv:
        await db.create_conversation(conversation_id, title=user_message[:50])

    image_result = None

    if file_bytes and filename and file_type:
        if ingest.is_image_type(file_type):
            image_result = await ingest.ingest_image(file_bytes, filename, file_type, conversation_id)
        else:
            await ingest.ingest_file(file_bytes, filename, file_type, conversation_id)

    context = ""
    if not image_result:
        try:
            results = rag.query(user_message, conversation_id=conversation_id)
            if results:
                context = "\n\n---\n\n".join(r["document"] for r in results)
        except RuntimeError:
            pass

    history = await db.get_messages(conversation_id)
    history = history[-MAX_HISTORY_MESSAGES:]

    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    if context:
        messages.append({"role": "system", "content": f"Relevant context from uploaded files:\n{context}"})

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    full_response = ""
    stream_error = None

    try:
        if image_result:
            async for token in model.generate_with_image(text=user_message, image_b64=image_result.image_b64, media_type=image_result.media_type):
                full_response += token
                yield token
        else:
            messages.append({"role": "user", "content": user_message})
            async for token in _run_with_tool_loop(messages):
                full_response += token
                yield token
        await db.insert_message(conversation_id, "user", user_message, has_file=bool(file_bytes), has_image=bool(image_result))
        await db.insert_message(conversation_id, "assistant", full_response)
        await db.update_conversation(conversation_id)
    except Exception as e:
        stream_error = str(e)
        if not full_response:
            full_response = f"Error: {stream_error}"
            yield f"\n[Error: {stream_error}]\n"


async def _run_with_tool_loop(messages: list[dict]) -> AsyncGenerator[str, None]:
    for round_idx in range(MAX_TOOL_ROUNDS):
        response_text = ""
        async for token in model.generate(messages=messages, tools=TOOLS):
            response_text += token
            yield token

        tool_calls = model.get_last_tool_calls()
        if not tool_calls:
            return

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            yield f"\n[calling {name}...]\n"
            result = await call_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result
            })

    yield "\n[max tool rounds reached]"

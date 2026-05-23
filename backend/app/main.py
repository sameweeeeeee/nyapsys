from contextlib import asynccontextmanager
import json
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app import db, rag, agent
from app.health import router as health_router
from app.schemas import IngestResponse, ConversationResponse, MessageResponse
from app.tools import TOOLS, call_tool


API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")


class ChatCompletionRequest(BaseModel):
    model: str = "nyapsys"
    messages: list[dict]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = True
    tools: Optional[list] = None


class FeedbackRequest(BaseModel):
    message_id: str
    score: float
    feedback: Optional[str] = None


async def verify_auth(request: Request):
    if request.url.path == "/health":
        return
    auth = request.headers.get("Authorization", "")
    if not API_SECRET_KEY:
        return
    if not auth.startswith("Bearer ") or auth[7:] != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    rag.init_collection()
    yield


app = FastAPI(title="Nyapsys Backend", lifespan=lifespan)
app.include_router(health_router, dependencies=[Depends(verify_auth)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", dependencies=[Depends(verify_auth)])
async def chat(message: str = Form(...), conversation_id: str = Form(...), file: UploadFile = File(None)):
    if not message.strip() and not file:
        raise HTTPException(status_code=400, detail="Message or file required")

    file_bytes = None
    filename = None
    file_type = None

    if file:
        file_bytes = await file.read()
        filename = file.filename
        file_type = filename.split(".")[-1] if "." in filename else "txt"

    async def event_stream():
        async for token in agent.run(user_message=message, conversation_id=conversation_id,
                                      file_bytes=file_bytes, filename=filename, file_type=file_type):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/chat/completions", dependencies=[Depends(verify_auth)])
async def v1_chat_completions(req: ChatCompletionRequest):
    messages = req.messages
    max_tokens = req.max_tokens or 2048
    temperature = req.temperature or 0.7

    if req.stream:
        async def event_stream():
            full_response = ""
            async for token in agent._stream_from_model(messages, max_tokens=max_tokens, temperature=temperature):
                full_response += token
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        content = ""
        async for token in agent._stream_from_model(messages, max_tokens=max_tokens, temperature=temperature):
            content += token
        return JSONResponse({"model": req.model, "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_auth)])
async def ingest_file(file: UploadFile = File(...), conversation_id: str = Form(...)):
    file_bytes = await file.read()
    filename = file.filename
    file_type = filename.split(".")[-1] if "." in filename else "txt"

    from app import ingest as ingest_module
    if ingest_module.is_image_type(file_type):
        result = await ingest_module.ingest_image(file_bytes, filename, file_type, conversation_id)
        return IngestResponse(file_id=result.file_id, chunk_count=0, filename=result.filename)
    else:
        result = await ingest_module.ingest_file(file_bytes, filename, file_type, conversation_id)
        return IngestResponse(file_id=result.file_id, chunk_count=result.chunk_count, filename=result.filename)


@app.get("/conversations", response_model=list[ConversationResponse], dependencies=[Depends(verify_auth)])
async def list_conversations():
    convs = await db.list_conversations()
    return [ConversationResponse(id=c["id"], title=c.get("title"), message_count=c.get("message_count", 0), updated_at=c["updated_at"]) for c in convs]


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse], dependencies=[Depends(verify_auth)])
async def get_messages(conversation_id: str):
    messages = await db.get_messages(conversation_id)
    return [MessageResponse(id=m["id"], role=m["role"], content=m["content"], has_file=m.get("has_file", False),
                             has_image=m.get("has_image", False), created_at=m["created_at"]) for m in messages]


@app.delete("/conversations/{conversation_id}", dependencies=[Depends(verify_auth)])
async def delete_conversation(conversation_id: str):
    rag.delete_by_conversation(conversation_id)
    await db.delete_conversation(conversation_id)
    return {"status": "deleted"}


@app.get("/v1/tools", dependencies=[Depends(verify_auth)])
async def get_tools():
    return {"tools": TOOLS}


@app.post("/v1/tools/call", dependencies=[Depends(verify_auth)])
async def invoke_tool(name: str = Form(...), args: str = Form(...)):
    import json
    result = await call_tool(name, json.loads(args))
    return {"result": result}


@app.post("/feedback", dependencies=[Depends(verify_auth)])
async def submit_feedback(req: FeedbackRequest):
    await db.insert_eval(req.message_id, req.score, req.feedback)
    return {"status": "recorded"}


import subprocess
import shlex


@app.get("/v1/training/logs", dependencies=[Depends(verify_auth)])
async def get_training_logs(lines: int = 500):
    import asyncio
    ssh_cmd = shlex.split(
        'gcloud compute ssh nyapsys-spot --zone=us-central1-a --tunnel-through-iap'
        f' --command="tail -{lines} /tmp/train.log"'
        ' -- -o ConnectTimeout=10 -o ServerAliveInterval=5'
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            return {"log": "", "parsed": [], "error": f"SSH failed (code {proc.returncode})", "connected": False}
        raw = stdout.decode(errors="replace")
        parsed = _parse_training_log(raw)
        return {"log": raw, "parsed": parsed, "error": None, "connected": True, "line_count": len(parsed)}
    except asyncio.TimeoutError:
        return {"log": "", "parsed": [], "error": "SSH timed out (30s)", "connected": False}
    except FileNotFoundError:
        return {"log": "", "parsed": [], "error": "gcloud not found on this machine", "connected": False}
    except Exception as e:
        return {"log": "", "parsed": [], "error": str(e), "connected": False}


def _parse_training_log(raw: str) -> list[str]:
    import re
    lines = raw.split("\r\n")
    if len(lines) == 1:
        lines = raw.split("\n")

    result: list[str] = []
    token_count = 0
    tqdm_buffer: list[str] = []
    MAX_TQDM = 12  # keep at most this many tqdm lines

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # skip raw tokenization progress
        if re.match(r'^\s*Tokenized [\d,]+ samples\.\.\.', stripped):
            token_count += 1
            continue

        # skip cudagraph warnings
        if 'torch/_inductor/cudagraph_utils' in stripped or '__cudagraphs' in stripped:
            continue

        # skip merging/saved noise
        if stripped.startswith('Merging ') or stripped.startswith('  Saved '):
            continue

        # tqdm lines — keep rolling buffer of last N
        if stripped.startswith('Training:'):
            # strip the progress bar characters to save space
            clean = re.sub(
                r'Training:\s+\d+%\|[^|]+\|\s+',
                '',
                stripped
            )
            tqdm_buffer.append(clean)
            if len(tqdm_buffer) > MAX_TQDM:
                tqdm_buffer.pop(0)
            continue

        result.append(stripped)

    if tqdm_buffer:
        for t in tqdm_buffer:
            result.append(t)

    if token_count:
        result.append(f"── Tokenization: {token_count} progress lines filtered ──")

    return result

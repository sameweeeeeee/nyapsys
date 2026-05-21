import os
import io
import uuid
import base64
from datetime import datetime
from dataclasses import dataclass

import fitz
from docx import Document
from PIL import Image
import pandas as pd
from app import db, rag


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
MIN_CHUNK_TOKENS = 50


@dataclass
class IngestResult:
    file_id: str
    chunk_count: int
    filename: str


@dataclass
class ImageResult:
    file_id: str
    image_b64: str
    media_type: str
    filename: str


def is_image_type(file_type: str) -> bool:
    return file_type.lower() in ["jpg", "jpeg", "png", "webp", "gif"]


def extract_text(file_bytes: bytes, file_type: str) -> str:
    if file_type == "pdf":
        return "\n\n".join([page.get_text() for page in fitz.open(stream=file_bytes, doc_type="pdf")])
    elif file_type == "docx":
        return "\n".join([p.text for p in Document(io.BytesIO(file_bytes)).paragraphs])
    elif file_type in ["txt", "md"]:
        return file_bytes.decode("utf-8", errors="replace")
    elif file_type == "csv":
        return pd.read_csv(io.BytesIO(file_bytes)).to_string(index=False)
    elif file_type == "json":
        import json
        return json.dumps(json.loads(file_bytes), indent=2)
    return file_bytes.decode("utf-8", errors="replace")


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if len(chunk.split()) >= MIN_CHUNK_TOKENS:
            chunks.append(chunk)
    return chunks


async def ingest_file(file_bytes: bytes, filename: str, file_type: str, conversation_id: str) -> IngestResult:
    text = extract_text(file_bytes, file_type)
    chunks = chunk_text(text)
    file_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    if chunks:
        ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"file_id": file_id, "filename": filename, "file_type": file_type, "chunk_index": i,
                      "total_chunks": len(chunks), "conversation_id": conversation_id, "created_at": created_at} for i in range(len(chunks))]
        rag.embed_and_upsert(chunks, metadatas, ids)
    else:
        ids = []

    await db.insert_file(file_id, conversation_id, filename, file_type, len(file_bytes), len(chunks), ids)
    return IngestResult(file_id, len(chunks), filename)


async def ingest_image(file_bytes: bytes, filename: str, file_type: str, conversation_id: str) -> ImageResult:
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((1120, 1120), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    file_id = str(uuid.uuid4())
    await db.insert_file(file_id, conversation_id, filename, file_type, len(file_bytes), 0, [])
    media_type = "image/jpeg"
    if file_type.lower() == "png": media_type = "image/png"
    elif file_type.lower() == "webp": media_type = "image/webp"
    elif file_type.lower() == "gif": media_type = "image/gif"
    return ImageResult(file_id, image_b64, media_type, filename)

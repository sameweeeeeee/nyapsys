import io
import uuid
import base64
from datetime import datetime
from typing import Optional

import fitz
from docx import Document
from PIL import Image
import pandas as pd


from app import db, rag


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
MIN_CHUNK_TOKENS = 50


try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
except Exception:
    TOKENIZER = None


import os


class IngestResult:
    def __init__(self, file_id: str, chunk_count: int, filename: str):
        self.file_id = file_id
        self.chunk_count = chunk_count
        self.filename = filename


class ImageIngestResult:
    def __init__(self, file_id: str, image_b64: str, media_type: str):
        self.file_id = file_id
        self.image_b64 = image_b64
        self.media_type = media_type


def is_image_type(file_type: str) -> bool:
    return file_type.lower() in ["jpg", "jpeg", "png", "webp", "gif"]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, doc_type="pdf")
    text = "\n\n".join([page.get_text() for page in doc])
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join([p.text for p in doc.paragraphs])
    return text


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def extract_text_from_csv(file_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_string(index=False)


def extract_text_from_json(file_bytes: bytes) -> str:
    import json

    data = json.loads(file_bytes)
    return json.dumps(data, indent=2)


def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []

    if TOKENIZER:
        tokens = TOKENIZER.encode(text, add_special_tokens=False)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + CHUNK_SIZE
            chunk_tokens = tokens[start:end]

            decoded = TOKENIZER.decode(chunk_tokens, skip_special_tokens=True)

            chunks.append(decoded.strip())

            start += CHUNK_SIZE - CHUNK_OVERLAP

        return [c for c in chunks if len(TOKENIZER.encode(c)) >= MIN_CHUNK_TOKENS]
    else:
        sentences = text.replace("\n", ". ").split(". ")
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) < CHUNK_SIZE * 4:
                current += sentence + ". "
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + ". "

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if len(c) >= MIN_CHUNK_TOKENS * 4]


async def ingest_file(
    file_bytes: bytes,
    filename: str,
    file_type: str,
    conversation_id: str,
) -> IngestResult:
    if file_type == "pdf":
        text = extract_text_from_pdf(file_bytes)
    elif file_type == "docx":
        text = extract_text_from_docx(file_bytes)
    elif file_type in ["txt", "md"]:
        text = extract_text_from_txt(file_bytes)
    elif file_type == "csv":
        text = extract_text_from_csv(file_bytes)
    elif file_type == "json":
        text = extract_text_from_json(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    chunks = chunk_text(text)

    file_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    if chunks:
        ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "file_id": file_id,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "conversation_id": conversation_id,
                "created_at": created_at,
            }
            for i in range(len(chunks))
        ]

        rag.embed_and_upsert(chunks, metadatas, ids)
    else:
        ids = []
        metadatas = []

    chroma_ids = ids if ids else []
    await db.insert_file(
        file_id,
        conversation_id,
        filename,
        file_type,
        len(file_bytes),
        len(chunks),
        chroma_ids,
    )

    return IngestResult(file_id, len(chunks), filename)


async def ingest_image(
    file_bytes: bytes,
    filename: str,
    file_type: str,
    conversation_id: str,
) -> ImageIngestResult:
    img = Image.open(io.BytesIO(file_bytes))

    if img.mode != "RGB":
        img = img.convert("RGB")

    max_size = 1120
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buffer.getvalue()).decode()

    file_id = str(uuid.uuid4())

    await db.insert_file(
        file_id,
        conversation_id,
        filename,
        file_type,
        len(file_bytes),
        0,
        [],
    )

    media_type = "image/jpeg"
    if file_type.lower() in ["png"]:
        media_type = "image/png"
    elif file_type.lower() in ["webp"]:
        media_type = "image/webp"
    elif file_type.lower() in ["gif"]:
        media_type = "image/gif"

    return ImageIngestResult(file_id, image_b64, media_type)
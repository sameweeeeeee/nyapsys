import os
import json
import uuid
from typing import Optional
import aiosqlite


SQLITE_PATH = os.getenv("SQLITE_PATH", "~/volumes/sqlite/nyapsys.db").replace("~", os.path.expanduser("~"))


async def init_db():
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0)
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, has_file BOOLEAN DEFAULT FALSE, has_image BOOLEAN DEFAULT FALSE,
                tokens_used INTEGER, latency_ms INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY, conversation_id TEXT, filename TEXT NOT NULL,
                file_type TEXT NOT NULL, file_size_bytes INTEGER, chunk_count INTEGER DEFAULT 0,
                chroma_ids TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS evals (
                id TEXT PRIMARY KEY, message_id TEXT, score REAL CHECK(score BETWEEN 0.0 AND 1.0),
                feedback TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        await db.commit()


async def create_conversation(conversation_id: str, title: Optional[str] = None):
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (conversation_id, title))
        await db.commit()
    return {"id": conversation_id, "title": title}


async def list_conversations():
    async with aiosqlite.connect(SQLITE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT c.*, COUNT(m.id) as message_count FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id ORDER BY c.updated_at DESC
        """) as cursor:
            return [dict(row) for row in await cursor.fetchall()]


async def get_messages(conversation_id: str):
    async with aiosqlite.connect(SQLITE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT id, role, content, has_file, has_image, created_at FROM messages
            WHERE conversation_id = ? ORDER BY created_at ASC
        """, (conversation_id,)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]


async def insert_message(conversation_id: str, role: str, content: str, has_file=False, has_image=False, tokens_used=None, latency_ms=None):
    message_id = str(uuid.uuid4())
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute("""
            INSERT INTO messages (id, conversation_id, role, content, has_file, has_image, tokens_used, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (message_id, conversation_id, role, content, has_file, has_image, tokens_used, latency_ms))
        await db.commit()
    return message_id


async def update_conversation(conversation_id: str):
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute("""
            UPDATE conversations SET updated_at = CURRENT_TIMESTAMP,
            message_count = (SELECT COUNT(*) FROM messages WHERE conversation_id = ?) WHERE id = ?
        """, (conversation_id, conversation_id))
        await db.commit()


async def delete_conversation(conversation_id: str):
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        await db.execute("DELETE FROM files WHERE conversation_id = ?", (conversation_id,))
        await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        await db.commit()


async def insert_file(file_id: str, conversation_id: str, filename: str, file_type: str, file_size_bytes: int, chunk_count: int, chroma_ids: list):
    async with aiosqlite.connect(SQLITE_PATH) as db:
        await db.execute("""
            INSERT INTO files (id, conversation_id, filename, file_type, file_size_bytes, chunk_count, chroma_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (file_id, conversation_id, filename, file_type, file_size_bytes, chunk_count, json.dumps(chroma_ids)))
        await db.commit()
    return {"id": file_id, "filename": filename}
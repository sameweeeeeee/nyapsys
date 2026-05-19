from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class MessageCreate(BaseModel):
    role: str
    content: str
    has_file: bool = False
    has_image: bool = False
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    has_file: bool = False
    has_image: bool = False
    created_at: datetime


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str] = None
    message_count: int = 0
    updated_at: datetime


class FileResponse(BaseModel):
    id: str
    conversation_id: Optional[str] = None
    filename: str
    file_type: str
    file_size_bytes: Optional[int] = None
    chunk_count: int = 0
    created_at: datetime


class FeedbackCreate(BaseModel):
    message_id: str
    score: float
    feedback: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    file: Optional[bytes] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None


class IngestResponse(BaseModel):
    file_id: str
    chunk_count: int
    filename: str


class HealthResponse(BaseModel):
    status: str
    model: str
    uptime_seconds: int
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    has_file: bool = False
    has_image: bool = False
    created_at: datetime


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str] = None
    message_count: int = 0
    updated_at: datetime


class IngestResponse(BaseModel):
    file_id: str
    chunk_count: int
    filename: str


class HealthResponse(BaseModel):
    status: str
    model: str
    uptime_seconds: int


class ToolCall(BaseModel):
    name: str
    args: dict


class ToolResponse(BaseModel):
    result: str
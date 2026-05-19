import time
import os

from fastapi import APIRouter
from app.schemas import HealthResponse


router = APIRouter()

START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model="loaded",
        uptime_seconds=int(time.time() - START_TIME),
    )
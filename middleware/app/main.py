from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import verify_token
from app.limiter import limiter
from app.health import router as health_router
from app.router import router as proxy_router


app = FastAPI(title="Nyapsys Middleware")

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)

app.mount("/", proxy_router)


from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import _limiter
from slowapi.errors import RateLimitExceeded


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Max 60 requests per minute."}
    )


@app.middleware("http")
async def add_rate_limit_header(request: Request, call_next):
    response = await call_next(request)
    return response
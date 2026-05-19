import os
from fastapi import HTTPException, Header


API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")


async def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = authorization.removeprefix("Bearer ")
    if token != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return True
import os
import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse


BACKEND_URL = os.getenv("BACKEND_URL", "http://DROPLET_PRIVATE_IP:8000")


@router.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    url = f"{BACKEND_URL}/{path}"
    
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    if request.method in ["POST", "PATCH"] and request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        files = {}
        data = {}
        
        for key, value in form.items():
            if hasattr(value, "filename"):
                files[key] = (value.filename, await value.read(), value.headers.get("content-type", "application/octet-stream"))
            else:
                data[key] = value
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    files=files if files else None,
                    data=data if data and not files else None,
                    timeout=300.0,
                )
            except httpx.ReadTimeout:
                return Response(content="Backend timeout", status_code=504)
            except httpx.ConnectError:
                return Response(content="Backend unavailable", status_code=503)
    else:
        body = await request.body()
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=body,
                    timeout=300.0,
                )
            except httpx.ReadTimeout:
                return Response(content="Backend timeout", status_code=504)
            except httpx.ConnectError:
                return Response(content="Backend unavailable", status_code=503)
    
    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    headers = [(k, v) for k, v in response.headers.items() if k.lower() not in excluded_headers]
    
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        async def event_stream():
            async for chunk in response.aiter_bytes():
                yield chunk
        
        return StreamingResponse(event_stream(), status_code=response.status_code, headers=headers)
    
    return Response(content=response.content, status_code=response.status_code, headers=headers)


from fastapi import APIRouter
router = APIRouter()
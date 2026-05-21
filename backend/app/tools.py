import httpx
import os
import json
import subprocess
import tempfile
from datetime import datetime


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python snippet and return stdout",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to run"}},
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": "Read a file from the Mac filesystem by path",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute file path"}},
                "required": ["path"]
            }
        }
    },
]


async def call_tool(name: str, args: dict) -> str:
    if name == "get_current_time":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if name == "web_search":
        api_key = os.getenv("SERPAPI_KEY", "")
        if api_key:
            async with httpx.AsyncClient() as client:
                r = await client.get("https://serpapi.com/search", params={"q": args["query"], "api_key": api_key, "num": 5})
                results = r.json().get("organic_results", [])
                return "\n".join(f"{r['title']}: {r['snippet']}" for r in results[:3])
        return "Web search not configured. Set SERPAPI_KEY in .env"

    if name == "run_python":
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(args["code"])
            fname = f.name
        result = subprocess.run(["python3", fname], capture_output=True, text=True, timeout=10)
        return result.stdout or result.stderr

    if name == "read_local_file":
        path = args["path"]
        allowed = [os.path.expanduser("~"), "/volumes"]
        if not any(path.startswith(a) for a in allowed):
            return "Access denied: path outside allowed directories"
        try:
            with open(path) as f:
                return f.read(4000)
        except Exception as e:
            return f"Error reading file: {e}"

    return f"Unknown tool: {name}"
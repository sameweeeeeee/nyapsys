import os
import json
import asyncio
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
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(args["query"], max_results=5))
            if not results:
                return "No results found"
            return "\n".join(f"{r['title']}: {r['body']}" for r in results[:3])
        except ImportError:
            return "Web search unavailable. Install duckduckgo-search: pip install duckduckgo-search"
        except Exception as e:
            return f"Web search error: {e}"

    if name == "run_python":
        timeout = int(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(args["code"])
            fname = f.name
        proc = await asyncio.create_subprocess_exec(
            "python3.11", fname,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode() or stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: Python execution timed out after {timeout}s"

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

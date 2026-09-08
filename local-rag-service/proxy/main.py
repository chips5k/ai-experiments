import json
import logging
import sys
import urllib.request
import urllib.parse
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pathlib import Path
# Force unbuffered standard output logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

LLAMA_SERVER_URL = "http://llama-cpp-server:8080/v1/chat/completions"
SEARXNG_URL = "http://searxng:8080/search"

HERMES_SYSTEM_PROMPT = """You are a function-calling AI assistant.
You MUST ALWAYS use a tool to answer ANY question about named individuals, real-time data, or specific facts. Never rely on internal knowledge for names or facts.

# Tools
You have access to the following tools:
<tools>
[
  {
    "type": "function",
    "function": {
      "name": "search_google",
      "description": "Searches the web for up-to-date real-time information.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "The search query string"}
        },
        "required": ["query"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_todos",
      "description": "Retrieves the system level todo list the user is maintaining.",
    }
  }
  {
    "type": "function",
    "function": {
      "name": "read_todo_detail",
      "description": "Reads detail file for a todo item",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {"type": "string", "description": "the custom name suffix for your detail file e.g file_<your_name>.md, should be uniquely identifiable from the top level todo entries"}
        },
        "required": ["name"]
      }
    }
  }
  {
    "type": "function",
    "function": {
      "name": "write_todos",
      "description": "writes a top level todo entry",
      "parameters": {
        "type": "object",
        "properties": {
          "data": {"type": "string", "description": "the full content you want to overwrite in the todos index file"}
        },
        "required": ["data"]
      }
    }
  }
  {
    "type": "function",
    "function": {
      "name": "write_todo_detail",
      "description": "writes supporting detailed data for top level todo entry",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {"type": "string", "description": "the custom name suffix for your detail file e.g file_<your_name>.md, should be uniquely identifiable from the top level todo entries"}
          "data": {"type": "string", "description": "the full content you want to overwrite in the todos index file"}
        },
        "required": ["data"]
      }
    }
  }
]
</tools>

To call a function, respond strictly and exclusively with:
<tool_call>
{"name": "<tool_name>", "arguments": {<args>}}
</tool_call>
ENSURE YOU INCLUDE the <tool_call>...</tool_call> wrapper
"""

def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}", flush=True)

def _check_searxng() -> bool:
    try:
        # Use health endpoint or minimal local query instead of hitting external search engines
        url = f"{SEARXNG_URL}?q=ping&format=json"
        req = urllib.request.Request(
            url, 
            headers={
                "User-Agent": "Mozilla/5.0",
                "X-Forwarded-For": "127.0.0.1"
            }
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        log(f"[STARTUP TEST] SearXNG check failed: {type(e).__name__} - {e}", "ERROR")
        return False

def _check_llama() -> bool:
    try:
        payload = json.dumps({"messages": [{"role": "user", "content": "ping"}]}).encode()
        req = urllib.request.Request(
            LLAMA_SERVER_URL, 
            data=payload, 
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        log(f"[STARTUP TEST] llama-cpp-server check failed: {type(e).__name__} - {e}", "ERROR")
        return False

def _sync_http_post(url: str, json_data: dict, timeout: float = 120.0) -> tuple[int, dict]:
    data = json.dumps(json_data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        res_body = e.read().decode("utf-8")
        try:
            parsed = json.loads(res_body)
        except Exception:
            parsed = {"error": res_body}
        return e.code, parsed

def _sync_searxng_get(query: str, timeout: float = 10.0) -> str:
    url = f"{SEARXNG_URL}?q={urllib.parse.quote(query)}&format=json"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "X-Forwarded-For": "127.0.0.1"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                return f"SearXNG error: HTTP status {response.status}"
            data = json.loads(response.read().decode("utf-8"))
            results = data.get("results", [])[:3]
            if not results:
                return "No relevant results found."
            return "\n".join([f"- {r.get('title')}: {r.get('content')}" for r in results])
    except Exception as e:
        log(f"[PROXY ERROR] SearXNG fetch failed: {type(e).__name__}: {e}", "ERROR")
        return "SearXNG search failed due to network issue."

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("=== WAITING FOR UPSTREAM SERVICES TO BE READY ===")
    
    # 1. Wait for llama-cpp-server health endpoint
    llama_ready = False
    for i in range(30):  # Retry up to 60 seconds (30 * 2s)
        if await asyncio.to_thread(_check_llama):
            llama_ready = True
            log("llama-cpp-server is healthy and ready.")
            break
        log(f"Waiting for llama-cpp-server... attempt {i+1}/30")
        await asyncio.sleep(2)

    if not llama_ready:
        log("llama-cpp-server failed to become healthy in time.", "ERROR")

    # 2. Check SearXNG
    searxng_ok = await asyncio.to_thread(_check_searxng)
    log(f"SearXNG status: {'OK' if searxng_ok else 'FAILED'}")

    if llama_ready and searxng_ok:
        log("=== ALL UPSTREAM SERVICES READY ===")
    else:
        log("=== STARTUP COMPLETED WITH UPSTREAM ISSUES ===", "WARN")

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    log("Health check requested")
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body payload"})

    log("Received /v1/chat/completions payload")

    messages = body.get("messages", [])
    
    # Prepend Hermes native XML tool system prompt if no system prompt exists
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": HERMES_SYSTEM_PROMPT})
    body["messages"] = messages

    # Step 1: Query llama-cpp-server
    log("Forwarding Step 1 request to llama-cpp-server")
    status_code, data = await asyncio.to_thread(_sync_http_post, LLAMA_SERVER_URL, body)
    if status_code != 200:
        log(f"[PROXY ERROR] llama-cpp-server returned status {status_code}", "ERROR")
        return JSONResponse(status_code=status_code, content=data)

    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError):
        log("[PROXY ERROR] Malformed payload from llama-cpp-server", "ERROR")
        return JSONResponse(status_code=500, content={"error": "Malformed response payload"})

    tool_calls = message.get("tool_calls")
    content = message.get("content", "") or ""

    # Parse XML <tool_call> tags from text content if native tool_calls are absent
    if not tool_calls and "<tool_call>" in content:
        log("Found <tool_call> tag in content, parsing...")
        try:
            start = content.find("<tool_call>") + len("<tool_call>")
            end = content.find("</tool_call>")
            raw_json = content[start:end].strip() if end != -1 else content[start:].strip()
            
            call_data = json.loads(raw_json)
            
            # Handle arguments whether already dict or raw JSON string
            args = call_data.get("arguments", {})
            args_str = json.dumps(args) if isinstance(args, dict) else str(args)

            tool_calls = [{
                "id": "call_1",
                "function": {
                    "name": call_data.get("name"),
                    "arguments": args_str
                }
            }]
        except Exception as e:
            log(f"[PROXY WARN] Failed to parse XML tool_call payload: {e}", "WARN")

    # Step 2: Intercept tool calls
    if tool_calls:
        log(f"Tool call detected: {len(tool_calls)} call(s)")
        messages.append(message)

        for call in tool_calls:
            fn_name = call.get("function", {}).get("name")
            raw_args = call.get("function", {}).get("arguments", "{}")
            
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}

            if fn_name == "search_google":
                query = args.get("query", "")
                log(f"Executing SearXNG tool search query: '{query}'")
                
                search_result = await asyncio.to_thread(_sync_searxng_get, query)
                
                # Append tool response in Hermes tag format
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", "call_1"),
                    "name": fn_name,
                    "content": f"<tool_response>\n{search_result}\n</tool_response>"
                })

            if fn_name == "list_todos":
                log(f"Loading users todos")
                
                path = Path("/custom/todo.md")
                content = await asyncio.to_thread(
                    path.read_text, encoding="utf-8"
                ) if path.exists() else ""

                # Append tool response in Hermes tag format
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", "call_1"),
                    "name": fn_name,
                    "content": f"<tool_response>\n{content}\n</tool_response>"
                })
            
            if fn_name == "read_todo_detail":
                log("Loading todo detail")
                
                name = args.get("name", "")
                target_path = Path(f"/custom/todo_{name}.md")
                log(f"reading todo detail: {target_path}")
                
                custom_dir = Path("/custom")
                dir_contents = await asyncio.to_thread(
                    lambda: [p.name for p in custom_dir.iterdir()] if custom_dir.exists() else []
                )
                
                log(f"Directory contents of /custom: {dir_contents}")
                
                if target_path.exists():
                    file_text = await asyncio.to_thread(target_path.read_text, encoding="utf-8")
                    content = f"Found File Contents for {target_path.name}: {file_text}"
                else:
                    log(f"File not found: {target_path}")
                    content = f"Failed to locate file: {target_path}\nDirectory contents of /custom: {dir_contents}"

                # Log incoming payload/context for debugging
                log(f"Tool call details -> ID: {call.get('id')}, args: {args}")

                # Append tool response in Hermes tag format
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", "call_1"),
                    "name": fn_name,
                    "content": f"<tool_response>\n{content}\n</tool_response>"
                })
            
            if fn_name == "write_todos":
                data = args.get("data", "")
                log("writing todo item")
                
                path = Path("/custom/todo.md")
                await asyncio.to_thread(
                    path.write_text, data, encoding="utf-8"
                )

                # Append tool response in Hermes tag format
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", "call_1"),
                    "name": fn_name,
                    "content": "<tool_response>\ntop level todo list updated\n</tool_response>"
                })
            
            if fn_name == "write_todo_detail":
                name = args.get("name", "")
                data = args.get("data", "")
                log(f"writing todo detail: {name}")
                
                path = Path(f"/custom/todo_{name}.md")
                await asyncio.to_thread(
                    path.write_text, data, encoding="utf-8"
                )

                # Append tool response in Hermes tag format
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", "call_1"),
                    "name": fn_name,
                    "content": f"<tool_response>\ntodo detail updated: todo_{name}.md\n</tool_response>"
                })

        # Step 3: Resubmit chat context for final text generation
        log("Resubmitting context with tool outputs to llama-cpp-server (Step 3)")
        body["messages"] = messages

        final_status, final_data = await asyncio.to_thread(_sync_http_post, LLAMA_SERVER_URL, body)
        if final_status != 200:
            log(f"[PROXY ERROR] Secondary request returned status {final_status}", "ERROR")
            return JSONResponse(status_code=final_status, content=final_data)
            
        return JSONResponse(content=final_data)

    log("No tool calls triggered; returning base LLM output")
    return JSONResponse(content=data)
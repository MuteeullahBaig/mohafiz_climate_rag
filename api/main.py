"""Mohafiz FastAPI service: API endpoints + mounted Gradio UI on one port.

HF Spaces (Docker SDK) exposes a single port; mounting Gradio onto FastAPI serves
the chat UI at "/" and the JSON/SSE API under "/api". Per-IP rate limiting (slowapi)
guards the API; the response cache + daily budget (agent/quota) guard the LLM spend.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from fastapi import FastAPI, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

import config
from agent import quota
from agent.graph import run_agent, stream_agent
from agent.tools import live_data
from ui.app import build_ui

limiter = Limiter(key_func=get_remote_address, default_limits=["60/hour"])
app = FastAPI(title="Mohafiz API", version="1.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _ratelimit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=429,
                        content={"error": "Rate limit exceeded. Please slow down."})


class Ask(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok", "budget": quota.budget_status(),
            "qdrant": "cloud" if config.QDRANT_URL else "local",
            "collection": config.COLLECTION_V2}


@app.get("/api/alerts")
def alerts():
    try:
        return live_data.get_gdacs_alerts()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.post("/api/ask")
@limiter.limit("10/hour")
def ask(request: Request, body: Ask):
    return run_agent(body.question)


@app.get("/api/ask/stream")
@limiter.limit("10/hour")
async def ask_stream(request: Request, question: str):
    def gen():
        for ev in stream_agent(question):
            yield {"data": json.dumps(ev, ensure_ascii=False)}
    return EventSourceResponse(gen())


# Mount the Gradio chat UI at "/" (must come after the API routes are declared).
app = gr.mount_gradio_app(app, build_ui(), path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

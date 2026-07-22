"""Hugging Face Gradio Space entrypoint.

The free HF tier hosts Gradio (not Docker) Spaces, so the Space launches the chat UI
directly rather than via FastAPI. The agent, retrieval (Qdrant Cloud), quota cache, and
daily-budget cap all work unchanged — set QDRANT_URL / QDRANT_API_KEY / GROQ_API_KEY as
Space secrets. (The FastAPI JSON+SSE service in api/main.py is for local/self-hosted runs.)
"""
from ui.app import build_ui

# queue() bounds concurrency — a lightweight rate-limit that protects the Groq quota
# alongside the cache + daily budget (the per-IP slowapi limiter is FastAPI-only).
demo = build_ui().queue(max_size=24, default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()

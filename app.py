"""Hugging Face Gradio Space entrypoint.

The free HF tier hosts ZeroGPU Gradio Spaces (Docker/CPU-Basic now need PRO), so the Space
launches the chat UI directly. ZeroGPU requires at least one @spaces.GPU function to exist
at startup, declared below. The agent's model inference (BGE-M3 + reranker) runs on CPU in
the main process — it fits comfortably in ZeroGPU's ~68GB RAM — while retrieval reads from
Qdrant Cloud and the LLM is Groq. Set QDRANT_URL / QDRANT_API_KEY / GROQ_API_KEY as secrets.
(The FastAPI JSON+SSE service in api/main.py is for local/self-hosted runs.)
"""
# `import spaces` must precede torch/transformers imports for ZeroGPU's fork server.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_probe():
        """Satisfies ZeroGPU's 'at least one @spaces.GPU function' startup check.
        Query inference runs on CPU in the main process, so this stays trivial."""
        import torch
        return torch.zeros(1).sum().item()
except ImportError:
    pass  # local dev: the `spaces` package is HF-Spaces-only

from ui.app import build_ui

# queue() bounds concurrency — a lightweight rate-limit protecting the Groq quota alongside
# the response cache + daily budget (the per-IP slowapi limiter is FastAPI-only).
demo = build_ui().queue(max_size=24, default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()

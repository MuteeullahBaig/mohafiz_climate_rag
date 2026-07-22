# Mohafiz — HF Spaces (Docker SDK) container. Serves FastAPI + mounted Gradio on 7860.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Run as the non-root user HF Spaces expects (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/hf_cache \
    PYTHONUNBUFFERED=1 \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding + reranker weights into the image so cold starts don't re-download
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('BAAI/bge-m3'); snapshot_download('BAAI/bge-reranker-v2-m3')" \
    && chown -R user:user /home/user/hf_cache

COPY --chown=user . .
# Writable runtime dir for the quota SQLite (cache + daily budget)
RUN mkdir -p /app/data && chown -R user:user /app

USER user
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

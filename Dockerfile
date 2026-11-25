# ---- RuleBook AI container ----
# One container = API + web UI + embedded Qdrant. Works on your laptop and on free hosting.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/app/models

WORKDIR /app

# 1) Install libraries first. Docker caches this layer, so code changes rebuild fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy our code
COPY app app
COPY scripts scripts

# 3) Download the embedding + reranker models at BUILD time, so the app starts fast.
ARG PRELOAD_MODELS=1
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
      python -c "from app.retrieval.embedder import FastEmbedder; from app.retrieval.reranker import Reranker; FastEmbedder(); Reranker()"; \
    fi

# 4) Documents to index on first start
COPY data data

# Hosting platforms often run as a non-root user -> make folders writable
RUN mkdir -p storage models && chmod -R 777 storage data models

EXPOSE 7860
# PORT is set by some hosts (Render); default 7860 (Hugging Face Spaces)
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

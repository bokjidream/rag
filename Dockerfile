FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

ARG TORCH_VERSION=2.8.0+cpu
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && pip install .

ARG PRELOAD_EMBEDDING_MODEL=false
RUN if [ "$PRELOAD_EMBEDDING_MODEL" = "true" ]; then \
        python -c "from sentence_transformers import SentenceTransformer; from src.embedding.kosimcse import MODEL_NAME; SentenceTransformer(MODEL_NAME)"; \
    fi

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

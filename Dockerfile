FROM python:3.9-slim

WORKDIR /app

# 시스템 의존성 (sentence-transformers, chromadb 빌드용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# pyproject.toml 먼저 복사 (레이어 캐시 최적화)
COPY pyproject.toml ./

# CPU 버전 torch 먼저 설치 (sentence-transformers가 GPU 버전 당기는 것 방지)
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# 소스 코드 복사
COPY src/ src/
COPY data/ data/

# 패키지 설치 (torch는 이미 설치됨)
RUN pip install --no-cache-dir -e .

# uvicorn이 PATH에 있는지 확인
RUN which uvicorn && uvicorn --version

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# 코딩 컨벤션

## 타입 힌트

모든 함수의 파라미터와 반환값에 타입 힌트 필수.

```python
# 올바른 예
async def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    ...

# 잘못된 예 (타입 힌트 없음)
async def search(query, top_k=5):
    ...
```

## async/await

FastAPI 엔드포인트와 Playwright 모두 비동기. sync 함수와 혼용하지 않는다.

```python
# 올바른 예
async def crawl_page(url: str) -> dict[str, str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        ...

# 잘못된 예 (sync Playwright 사용)
def crawl_page(url: str) -> dict[str, str]:
    with sync_playwright() as p:
        ...
```

## 에러 처리

### 크롤러: 실패해도 계속 진행

```python
async def crawl_urls(urls: list[str]) -> list[dict[str, Any]]:
    results = []
    for url in urls:
        try:
            data = await crawl_page(url)
            results.append(data)
        except Exception as e:
            logger.warning("크롤링 실패, 건너뜀: url=%s error=%s", url, e)
            continue  # 파이프라인 중단 금지
    return results
```

### 파이프라인: 청크 실패 시 skip

```python
async def embed_chunks(chunks: list[str]) -> list[list[float]]:
    embeddings = []
    for chunk in chunks:
        try:
            emb = await embed(chunk)
            embeddings.append(emb)
        except Exception as e:
            logger.warning("임베딩 실패, 청크 건너뜀: error=%s", e)
            continue  # 해당 청크만 skip
    return embeddings
```

### API: 일관된 에러 응답

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )
```

## ruff 설정 요약

`pyproject.toml` 기준:
- `line-length = 100`
- `target-version = "py39"`
- import 정렬: isort 호환

## 로깅

```python
import logging

logger = logging.getLogger(__name__)

# 사용
logger.info("작업 시작: %s", context)
logger.warning("경고: %s", detail)
logger.error("에러 발생: %s", error)
```

- `print()` 사용 금지 → `logger` 사용
- 민감 정보(URL 파라미터, 개인정보 등) 로그에 포함 금지

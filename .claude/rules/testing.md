---
glob: "**/tests/**,**/*_test.py"
---

# 테스트 규칙 — tests/ 작업 시 적용

## 커버리지

- 80% 이상 필수 (`pytest --cov=src --cov-fail-under=80`)

## 모듈별 테스트 전략

| 모듈 | 전략 |
|------|------|
| `src/crawler/` | fixture HTML 사용, 실제 네트워크 요청 금지 |
| `src/pipeline/` | 샘플 문서로 청킹·임베딩 단위 테스트 |
| `src/retriever/` | `FastAPI TestClient` + ChromaDB in-memory 컬렉션 |

## fixture HTML 예시 (crawler 테스트)

```python
@pytest.fixture
def sample_html() -> str:
    return "<html><body><h1>복지서비스</h1></body></html>"

async def test_parse_welfare(sample_html: str) -> None:
    result = parse_welfare_page(sample_html)
    assert result["service_name"] is not None
```

## ChromaDB in-memory 예시 (retriever 테스트)

```python
import chromadb

@pytest.fixture
def chroma_client() -> chromadb.Client:
    return chromadb.Client()  # in-memory, 테스트 후 자동 소멸
```

## 참고

테스트 작성 시 → [docs/conventions.md](../../docs/conventions.md)

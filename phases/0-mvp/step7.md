# Step 7: api-layer

## 읽어야 할 파일

먼저 아래 파일들을 읽고 설계 의도를 파악하라:

- `docs/PRD.md` (API 1, API 2 스펙 확인)
- `docs/ARCHITECTURE.md`
- `src/models/welfare.py` (SearchRequest, SearchResponse, WelfareDetail 확인)
- `src/retriever/search.py` (search_welfare, get_welfare_detail 시그니처 확인)
- `src/embedding/kosimcse.py` (KoSimCSEEmbedder 확인)
- `phases/0-mvp/index.json` (step 6 summary 확인)

## 작업

FastAPI 앱을 완성하고 LangGraph 팀이 호출하는 두 엔드포인트를 구현한다.

### 구현할 파일

#### 1. `src/api/deps.py` — 공통 의존성 (순환 임포트 방지)

```python
from __future__ import annotations
from fastapi import Request
from src.embedding.protocol import EmbedderProtocol

def get_embedder(request: Request) -> EmbedderProtocol:
    return request.app.state.embedder  # type: ignore[no-any-return]
```

- `main.py`가 `routes/welfare.py`를 임포트하고, `routes/welfare.py`가 다시 `main.py`를 임포트하면
  순환 임포트가 발생한다. `get_embedder`를 독립 모듈 `deps.py`에 두어 이를 방지한다.
- `routes/welfare.py`는 `src.api.deps`에서 `get_embedder`를 임포트한다.

#### 2. `src/api/main.py` — FastAPI 앱 초기화

```python
from __future__ import annotations
from fastapi import FastAPI
from src.api.routes.welfare import router

app = FastAPI(title="BokjiDream RAG API", version="0.1.0")
app.include_router(router)
```

- 앱 시작 시(`lifespan`) `KoSimCSEEmbedder` 인스턴스를 생성하여 `app.state.embedder`에 저장

#### 3. `src/api/routes/welfare.py` — 라우터

```python
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from src.api.deps import get_embedder
from src.embedding.protocol import EmbedderProtocol
from src.models.welfare import SearchRequest, SearchResponse, WelfareDetail
from src.retriever.search import search_welfare, get_welfare_detail

router = APIRouter(prefix="/welfare", tags=["welfare"])

@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    embedder: EmbedderProtocol = Depends(get_embedder),
) -> SearchResponse:
    """POST /welfare/search — 유저 조건으로 관련 서비스 top-k 검색."""
    if request.disability and request.disability_severity is None:
        raise HTTPException(status_code=422, detail="disability=True이면 disability_severity를 지정해야 합니다.")
    return await search_welfare(request, embedder)

@router.get("/{serv_id}", response_model=WelfareDetail)
async def get_detail(serv_id: str) -> WelfareDetail:
    """GET /welfare/{serv_id} — 서비스 상세 정보 반환.
    
    ChromaDB ID 조회만 수행하므로 embedder 의존성 없음.
    """
    detail = await get_welfare_detail(serv_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"서비스를 찾을 수 없습니다: {serv_id}")
    return detail
```

- `disability=True, disability_severity=None` → 422 (cross-field 검증, route handler에서 처리)
- `serv_id`가 존재하지 않으면 404 반환
- `get_detail`은 ChromaDB ID 조회만 하므로 embedder가 필요 없음 — `Depends(get_embedder)` 없음

### `get_embedder` 의존성 위치

`get_embedder`는 `src/api/deps.py`에 정의한다 (위 1번 항목 참조).
`routes/welfare.py`는 `from src.api.deps import get_embedder`로 임포트한다.

### lifespan 설정

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.embedding.kosimcse import KoSimCSEEmbedder

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedder = KoSimCSEEmbedder()
    yield
    # shutdown 시 정리 필요한 경우 여기에 추가

app = FastAPI(lifespan=lifespan, ...)
```

### 주의사항

- `api/`에서 `pipeline/`, `crawler/`를 임포트하지 마라
- `KoSimCSEEmbedder`는 `main.py` lifespan에서 1회만 인스턴스화
- `routes/welfare.py`는 `EmbedderProtocol` 타입만 참조 (구체 구현 숨김)

## Acceptance Criteria

```bash
mypy src/api/                                    # 타입 오류 없음
pytest tests/integration/test_api.py -v        # 모든 테스트 통과
ruff check src/api/                             # 린트 오류 없음
```

테스트 파일 `tests/integration/test_api.py`도 함께 작성한다.
`unittest.mock.patch`로 `src.retriever.search.search_welfare`와
`src.retriever.search.get_welfare_detail`을 mock 처리한다
(ChromaDB/임베더 실제 호출 불필요, API 레이어만 테스트).
최소한 아래 케이스를 커버해야 한다:

- `POST /welfare/search` 유효한 요청 → 200 + `SearchResponse` 형식 반환
- `POST /welfare/search` 필수 필드 누락 → 422
- `POST /welfare/search` `disability=True, disability_severity=None` → 422
- `GET /welfare/{serv_id}` 존재하는 ID → 200 + `WelfareDetail` 형식 반환
- `GET /welfare/{serv_id}` 없는 ID → 404

## 검증 절차

1. 위 AC 커맨드를 순서대로 실행한다.
2. 전체 파이프라인 통합 검증:
   ```bash
   uvicorn src.api.main:app --reload
   # 새 터미널에서:
   curl -X POST http://localhost:8000/welfare/search \
     -H "Content-Type: application/json" \
     -d '{"age": 65, "income_level": "저소득"}'
   ```
3. 아키텍처 체크리스트:
   - `api/`가 `pipeline/`, `crawler/`를 임포트하지 않는가?
   - `KoSimCSEEmbedder`가 lifespan에서 1회만 생성되는가?
   - PRD의 두 엔드포인트(`POST /welfare/search`, `GET /welfare/{serv_id}`)가 모두 구현되었는가?
4. 결과에 따라 `phases/0-mvp/index.json`의 step 7을 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "src/api/ 구현 완료 — FastAPI 앱, POST /welfare/search, GET /welfare/{serv_id} 엔드포인트 완료"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- `api/routes/welfare.py`에서 `KoSimCSEEmbedder`를 직접 import하지 마라 (lifespan DI 사용)
- `get_embedder`를 `main.py`에서 직접 import하지 마라 — 반드시 `deps.py` 경유 (순환 임포트 방지)
- `api/`에서 `crawler/`, `pipeline/`을 임포트하지 마라
- 사용자 개인정보(나이, 소득 등)를 응답에 포함하거나 로그에 출력하지 마라

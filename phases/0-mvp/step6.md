# Step 6: retriever

## 읽어야 할 파일

먼저 아래 파일들을 읽고 설계 의도를 파악하라:

- `docs/ARCHITECTURE.md`
- `docs/ADR.md` (ADR-006, ADR-007)
- `src/models/welfare.py` (SearchRequest, SearchResult, WelfareDetail 확인)
- `src/db/chroma.py` (get_collection, WELFARE_COLLECTION 확인)
- `src/embedding/protocol.py` (EmbedderProtocol 확인)
- `phases/0-mvp/index.json` (step 5 summary 확인)

## 작업

`src/retriever/search.py`를 구현한다.
`SearchRequest`를 자연어 쿼리로 변환 → 임베딩 → ChromaDB 유사도 검색 → `SearchResult` 리스트 반환.

### 구현할 파일

#### `src/retriever/search.py`

```python
from __future__ import annotations
import json
from src.models.welfare import SearchRequest, SearchResult, SearchResponse, WelfareDetail
from src.db.chroma import get_collection, WELFARE_COLLECTION
from src.embedding.protocol import EmbedderProtocol

def build_query_text(request: SearchRequest) -> str:
    """SearchRequest → 한국어 자연어 쿼리 문자열 변환."""

async def search_welfare(
    request: SearchRequest,
    embedder: EmbedderProtocol,
) -> SearchResponse:
    """RAG 검색 메인 함수. SearchRequest → SearchResponse."""

async def get_welfare_detail(serv_id: str) -> WelfareDetail | None:
    """ChromaDB에서 serv_id로 상세 정보 조회."""
```

### `build_query_text` 쿼리 생성 로직

유저 조건을 한국어 자연어 문장으로 변환하여 벡터 검색 품질을 높인다.
**아래 구현을 그대로 사용한다** (자의적 변형 금지):

```python
def build_query_text(request: SearchRequest) -> str:
    parts: list[str] = [f"{request.age}세", request.income_level]
    if request.household_size is not None:
        parts.append(f"{request.household_size}인 가구")
    if request.marital_status is not None:
        parts.append(request.marital_status)
    if request.has_children is True:
        parts.append("미성년 자녀 있음")
    if request.disability:
        if request.disability_severity:
            parts.append(f"장애인({request.disability_severity})")
        else:
            parts.append("장애인")
    if request.employment_status is not None:
        parts.append(request.employment_status)
    if request.region is not None:
        parts.append(request.region)
    return " ".join(parts) + " 거주자를 위한 복지 서비스"
```

필드별 렌더링 규칙:
- `age` → `"{age}세"` (항상 포함)
- `income_level` → 값 그대로 (항상 포함)
- `household_size` → `"{n}인 가구"` (not None일 때)
- `marital_status` → 값 그대로 (not None일 때)
- `has_children=True` → `"미성년 자녀 있음"`, `False/None` → 포함하지 않음
- `disability=True, disability_severity="중증"` → `"장애인(중증)"`
- `disability=True, disability_severity="경증"` → `"장애인(경증)"`
- `disability=True, disability_severity=None` → `"장애인"` (Step 7에서 422 처리되므로 실제론 미발생)
- `disability=False` → 포함하지 않음
- `employment_status` → 값 그대로 (not None일 때)
- `region` → 값 그대로 (not None일 때)
- 고정 suffix: `" 거주자를 위한 복지 서비스"`

예시 출력:
- `"65세 저소득 1인 가구 미혼 서울 거주자를 위한 복지 서비스"`
- `"45세 기초생활수급자 장애인(중증) 취업자 경기도 거주자를 위한 복지 서비스"`

### `search_welfare` 검색 로직

1. `build_query_text(request)` → 쿼리 문자열 생성
2. `vec = embedder.embed([query_text])[0]` → 쿼리 벡터 (embed는 list 반환, 첫 번째 요소 추출)
3. ChromaDB `query()` 호출: `n_results=request.top_k`
4. **ChromaDB `query()` 결과 구조 주의**:
   ```python
   result = await asyncio.to_thread(collection.query, query_embeddings=[vec], n_results=top_k)
   # result 구조: {"ids": [[...]], "distances": [[...]], "metadatas": [[...]]}
   # 쿼리 1개이므로 항상 [0] 인덱스로 접근
   ids        = result["ids"][0]        # list[str]
   distances  = result["distances"][0]  # list[float]
   metadatas  = result["metadatas"][0]  # list[dict]
   ```
5. 각 `(metadata, distance)` 쌍에서 `SearchResult` 필드 매핑:
   - `serv_id` = `metadata["serv_id"]`
   - `serv_nm` = `metadata["serv_nm"]`
   - `serv_dgst` = `metadata["serv_dgst"]`
   - `department` = `metadata["jur_mnof_nm"]`  ← **반드시 jur_mnof_nm 키 사용**
   - `score` = `max(0.0, 1.0 - distance)`
   - `trgter_indvdl` = `json.loads(metadata["trgter_indvdl"])`
   - `intrs_thema` = `json.loads(metadata["intrs_thema"])`
6. `SearchResult` 리스트 생성 및 `SearchResponse` 반환

> **메타데이터 사전 필터링 미적용 이유**: `trgter_indvdl`과 `intrs_thema`는 ChromaDB에
> JSON 문자열(`'["저소득","노인"]'`)로 저장된다. ChromaDB 1.5.x `where` 절은 문자열 내
> substring 검색(`$contains`)을 지원하지 않으므로 이 필드로 사전 필터링할 수 없다.
> MVP에서는 순수 시맨틱 검색(코사인 유사도)만으로 관련 서비스를 찾는다.
> 정밀도 향상이 필요하면 후처리 단계(Python 레벨)에서 `trgter_indvdl` 포함 여부를
> 체크하는 방식으로 확장한다.

### `get_welfare_detail` 상세 조회 로직

1. ChromaDB `get()` 호출로 `serv_id` 일치하는 청크 전체 조회:
   ```python
   result = await asyncio.to_thread(collection.get, where={"serv_id": {"$eq": serv_id}})
   ```
2. `result["ids"]`가 비어 있으면 `None` 반환
3. 첫 번째 항목의 `metadatas[0]`에서 `WelfareDetail` 전체 필드를 매핑한다:
   ```python
   meta = result["metadatas"][0]
   return WelfareDetail(
       serv_id        = meta["serv_id"],
       serv_nm        = meta["serv_nm"],
       serv_dgst      = meta["serv_dgst"],
       tgtr_dtl_cn    = meta["tgtr_dtl_cn"],
       slct_crit_cn   = meta["slct_crit_cn"],
       alw_serv_cn    = meta["alw_serv_cn"],
       sprt_cyc_nm    = meta["sprt_cyc_nm"],
       srv_pvsn_nm    = meta["srv_pvsn_nm"],
       trgter_indvdl  = json.loads(meta["trgter_indvdl"]),
       intrs_thema    = json.loads(meta["intrs_thema"]),
       application_url = meta["serv_dtl_link"],  # ← serv_dtl_link 키 사용
   )
   ```

### 주의사항

- `embedder`를 DI로 받는다 — retriever가 `KoSimCSEEmbedder`를 직접 import하지 마라
- `json.loads()`로 `trgter_indvdl`, `intrs_thema` 역직렬화 (pipeline에서 JSON 문자열로 저장됨)
- **chromadb 호출(`collection.query`, `collection.get`)은 반드시 `asyncio.to_thread()`로 래핑** (ADR-008):
  ```python
  result = await asyncio.to_thread(collection.query, query_embeddings=[vec], n_results=top_k)
  result = await asyncio.to_thread(collection.get, where={"serv_id": {"$eq": serv_id}})
  ```
- ChromaDB `query()` 결과 `distances`를 `score`로 변환:
  ```python
  # cosine distance 범위: [0, 2]. distance=0은 동일 벡터
  # score = 1 - distance 는 [-1, 1] 범위이므로 clamp 적용
  score = max(0.0, 1.0 - distance)
  ```
  Step 2에서 `hnsw:space: "cosine"`으로 컬렉션을 생성했기 때문에 이 공식이 유효하다.

## Acceptance Criteria

```bash
mypy src/retriever/                            # 타입 오류 없음
pytest tests/unit/test_search.py -v          # 모든 테스트 통과
ruff check src/retriever/                     # 린트 오류 없음
```

테스트 파일 `tests/unit/test_search.py`도 함께 작성한다.
최소한 아래 케이스를 커버해야 한다:

- `build_query_text()` 필수 필드만 → 최소 쿼리 생성
- `build_query_text()` 전체 필드 → 모든 조건 포함된 쿼리 생성
- `search_welfare()` mock embedder + mock ChromaDB → `SearchResponse` 반환 검증
- `get_welfare_detail()` 존재하는 `serv_id` → `WelfareDetail` 반환
- `get_welfare_detail()` 없는 `serv_id` → `None` 반환

## 검증 절차

1. 위 AC 커맨드를 순서대로 실행한다.
2. 아키텍처 체크리스트:
   - `retriever/`가 `api/`, `pipeline/`, `crawler/`를 임포트하지 않는가?
   - `embedder`를 DI로 받는가?
   - score 변환에 `max(0.0, 1.0 - distance)` clamp가 적용되었는가?
3. 결과에 따라 `phases/0-mvp/index.json`의 step 6을 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "src/retriever/search.py 구현 완료 — build_query_text, search_welfare, get_welfare_detail 함수 완료"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- `retriever/`에서 `api/`, `pipeline/`, `crawler/`를 임포트하지 마라
- `KoSimCSEEmbedder`를 직접 인스턴스화하지 마라 (DI 원칙)
- 사용자 개인정보(나이, 소득 등)를 외부 API에 전송하지 마라 (Sovereign AI)

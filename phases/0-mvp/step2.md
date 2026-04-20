# Step 2: db-layer

## 읽어야 할 파일

먼저 아래 파일들을 읽고 설계 의도를 파악하라:

- `docs/ARCHITECTURE.md`
- `docs/ADR.md` (ADR-001, ADR-008)
- `phases/0-mvp/index.json` (step 1 summary 확인)

## 작업

`src/db/chroma.py`를 생성한다. ChromaDB 클라이언트 싱글턴을 관리하는 유일한 모듈이다.
**프로젝트 전체에서 ChromaDB 클라이언트는 이 모듈에서만 초기화한다.**

### 배경 — chromadb 1.5.7 async 제약

chromadb 1.5.7에서 `AsyncEphemeralClient` / `AsyncPersistentClient`는 존재하지 않는다.
`AsyncHttpClient`는 원격 HTTP 서버 전용이므로 로컬 사용에 적합하지 않다.
**따라서 동기 클라이언트에 `asyncio.to_thread()`를 래핑하는 방식을 사용한다.**
이 결정은 ADR-008에 기록되어 있다.

### 구현할 내용

#### 1. ChromaDB 클라이언트 싱글턴

```python
# src/db/chroma.py
from __future__ import annotations

import asyncio
import os
import chromadb
from chromadb import ClientAPI

_client: ClientAPI | None = None
_lock: asyncio.Lock | None = None  # 모듈 레벨 초기화 금지 (Python 3.9 이벤트루프 바인딩 오류)

async def get_client() -> ClientAPI:
    """ChromaDB 클라이언트 싱글턴 반환. 앱 시작 시 최초 1회 초기화.

    CHROMA_MODE=ephemeral  → EphemeralClient (in-memory, 테스트용)
    CHROMA_MODE=persistent → PersistentClient (기본값, data/chroma 경로)
    asyncio.to_thread()로 래핑하여 이벤트 루프 블로킹 방지.

    asyncio.Lock 없이 동시 호출 시: await가 이벤트 루프를 양보하는 순간
    다른 코루틴이 _client is None 체크를 통과해 클라이언트를 중복 생성한다.
    EphemeralClient의 경우 서로 다른 빈 DB 인스턴스를 가리키게 되어
    인덱싱한 데이터가 검색에서 보이지 않는 버그가 발생한다.

    asyncio.Lock은 반드시 실행 중인 이벤트 루프 안에서 생성해야 한다 (Python 3.9).
    모듈 레벨에 두면 "attached to a different loop" RuntimeError가 발생한다.
    """
    global _client, _lock
    if _lock is None:
        _lock = asyncio.Lock()  # 이벤트 루프 실행 중에 생성 — Python 3.9 호환
    async with _lock:
        if _client is None:
            mode = os.getenv("CHROMA_MODE", "persistent")
            if mode == "ephemeral":
                _client = await asyncio.to_thread(chromadb.EphemeralClient)
            else:
                path = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
                _client = await asyncio.to_thread(chromadb.PersistentClient, path)
    return _client

async def get_collection(name: str) -> chromadb.Collection:
    """컬렉션 반환. 없으면 생성. 거리 메트릭은 cosine으로 고정."""
    client = await get_client()
    return await asyncio.to_thread(
        client.get_or_create_collection,
        name,
        metadata={"hnsw:space": "cosine"},  # retriever score = max(0, 1 - distance) 공식에 의존
    )
```

- 모든 chromadb 호출은 `asyncio.to_thread()`로 래핑한다
- `ClientAPI`는 `chromadb.ClientAPI` 타입을 사용한다 (`chromadb.AsyncClientAPI`가 아님)

#### 2. 컬렉션 이름 상수

```python
WELFARE_COLLECTION = "welfare_services"
```

### 환경변수

```
CHROMA_PERSIST_DIR=data/chroma   # 기본값
CHROMA_MODE=persistent           # "persistent" | "ephemeral"
```

### 주의사항

- `ClientAPI` 타입은 `chromadb` 패키지에서 임포트한다 (`AsyncClientAPI` 아님)
- 싱글턴이므로 테스트에서는 아래처럼 두 전역 변수를 함께 리셋한다:
  ```python
  import src.db.chroma as db_module
  db_module._client = None
  db_module._lock = None
  ```
- `src/models/`만 임포트 가능. 다른 레이어 임포트 금지
- chromadb cosine distance 범위는 [0, 2]. distance=0은 동일 벡터, distance=2는 반대 방향
- `pipeline/`과 `retriever/`에서 chromadb 연산 호출 시 반드시 `asyncio.to_thread()` 래핑

## Acceptance Criteria

```bash
mypy src/db/                              # 타입 오류 없음
pytest tests/unit/test_db.py -v          # 모든 테스트 통과
ruff check src/db/                        # 린트 오류 없음
```

테스트 파일 `tests/unit/test_db.py`도 함께 작성한다.
최소한 아래 케이스를 커버해야 한다:

- `get_client()` 두 번 호출 시 동일 인스턴스 반환 (싱글턴 검증)
- `get_client()` 동시 호출(asyncio.gather) 시 동일 인스턴스 반환 (race condition 없음)
- `CHROMA_MODE=ephemeral` 환경 설정 시 `EphemeralClient` 사용
- `CHROMA_MODE=persistent` 환경 설정 시 `PersistentClient` 사용
- `get_collection()` 호출 시 `hnsw:space=cosine` 메타데이터로 컬렉션 반환

## 검증 절차

1. 위 AC 커맨드를 순서대로 실행한다.
2. 아키텍처 체크리스트:
   - ChromaDB 클라이언트 초기화가 `src/db/chroma.py` 한 곳에만 있는가?
   - 모든 chromadb 호출이 `asyncio.to_thread()`로 래핑되었는가?
   - `get_or_create_collection`에 `hnsw:space: "cosine"`이 명시되었는가?
   - `WELFARE_COLLECTION` 상수가 정의되었는가?
3. 결과에 따라 `phases/0-mvp/index.json`의 step 2를 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "src/db/chroma.py 생성 — 동기 클라이언트+asyncio.to_thread 싱글턴(persistent/ephemeral 분기), cosine metric, WELFARE_COLLECTION 상수 완료"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- `src/db/` 외부 레이어(`crawler`, `pipeline`, `retriever`, `api`)를 임포트하지 마라
- ChromaDB 클라이언트를 여러 모듈에 분산하지 마라
- `chromadb.AsyncEphemeralClient()` / `chromadb.AsyncPersistentClient()`를 사용하지 마라 — 1.5.7에 존재하지 않음
- chromadb 호출 시 `asyncio.to_thread()` 래핑을 생략하지 마라

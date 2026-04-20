# Architecture Decision Records

## 철학

개인정보를 외부에 전송하지 않는 Sovereign AI 원칙 준수. 작동하는 최소 구현 우선, LangGraph 팀과의 인터페이스 안정성 최우선.

---

### ADR-001: 벡터 DB — ChromaDB 선택

**결정**: Pinecone, Weaviate 대신 ChromaDB 사용  
**이유**: 로컬 실행 가능 (개인정보 외부 전송 없음). 설치·설정 없이 Python 패키지만으로 동작. MVP 속도 우선  
**트레이드오프**: Pinecone 대비 대규모 벡터(수백만 건) 성능 열위. 클라우드 관리형 서비스 없음

---

### ADR-002: 임베딩 모델 — sentence-transformers 한국어 모델

**결정**: OpenAI Embedding API 대신 로컬 sentence-transformers 사용  
**이유**: 사용자 쿼리(개인 상황 정보 포함)를 외부 API에 전송하지 않아야 함 (Sovereign AI). 무료. 오프라인 동작  
**트레이드오프**: OpenAI text-embedding-3 대비 품질 열위 가능성. 모델 용량(수백MB) 로컬 저장 필요  
**후보 모델**: `jhgan/ko-sroberta-multitask` 또는 `snunlp/KR-ELECTRA-discriminator` (실험 후 결정)

---

### ADR-003: API 프레임워크 — FastAPI

**결정**: Flask, Django 대신 FastAPI 사용  
**이유**: 비동기(async) 지원으로 크롤링·DB 동시 처리 효율적. Pydantic 기반 자동 스키마 검증. LangGraph 팀과 API 스펙 공유 시 자동 생성 OpenAPI 문서 활용  
**트레이드오프**: 팀원이 Flask에 더 익숙할 수 있음

---

### ADR-004: 크롤링 — Playwright + httpx 병행

**결정**: Playwright(동적 페이지)와 httpx(정적 API)를 함께 사용  
**이유**: 복지로는 JavaScript 렌더링 필요 → Playwright. 공공데이터포털은 REST API → httpx로 충분  
**트레이드오프**: Playwright 설치·브라우저 바이너리 용량(~300MB). CI 환경 설정 필요

---

### ADR-005: 데이터 수집 전략 — 공공데이터 API 우선, 크롤링 후순위

**결정**: MVP에서는 공공데이터포털 복지서비스 API(목록 조회 + 상세 조회) 두 개만 사용. 크롤링은 추가하지 않음  
**이유**: 상세 API가 `tgtrDtlCn`(수급 대상), `slctCritCn`(선정 기준), `trgterIndvdlArray`(대상 분류) 등 RAG에 필요한 핵심 필드를 충분히 제공함. 크롤링 추가 시 복잡도 대비 품질 향상이 불확실  
**트레이드오프**: `basfrmList`의 HWP/PDF 업무처리지침에 소득·재산 기준 등 세부 숫자 조건이 있을 수 있음 — 검색 품질 검증 후 필요하면 PDF 파싱 추가  
**확장 계획**: `src/crawler/` 레이어는 미리 만들어두되 내부는 API 클라이언트만 구현. 크롤러 추가 시 같은 레이어에 확장

---

### ADR-006: 검색 API 입력 — 유저 프로파일 JSON (자연어 쿼리 아님)

**결정**: LangGraph가 자연어 쿼리 대신 유저 조건 구조체(나이, 소득 수준, 장애 여부 등)를 전달하고, RAG 레이어가 이를 벡터 검색 쿼리 + ChromaDB 메타데이터 필터로 변환. 엔드포인트는 `/welfare` 리소스 아래 일관성 있게 구성 (`POST /welfare/search`, `GET /welfare/{serv_id}`)  
**이유**: 관심사 분리 — LangGraph는 유저 인터페이스만, RAG는 검색 로직만 담당. Sovereign AI 원칙상 유저 개인정보를 외부 임베딩 API에 전송하지 않으므로 RAG가 직접 쿼리를 생성해야 함. `/welfare` 단일 리소스 네임스페이스로 REST 일관성 확보  
**확정 스펙**:
```python
class SearchRequest(BaseModel):
    age: int                          # 필수
    income_level: Literal["기초생활수급자", "차상위계층", "저소득", "일반"]  # 필수
    household_size: int | None = None
    marital_status: Literal["미혼", "기혼", "이혼", "사별"] | None = None
    has_children: bool | None = None
    disability: bool = False
    disability_severity: Literal["경증", "중증"] | None = None
    employment_status: Literal["취업", "실업", "비경제활동"] | None = None
    region: str | None = None
    top_k: int = 5
```
**트레이드오프**: 일반적인 RAG 패턴(쿼리 문자열 수신)과 다름 — LangGraph 팀과 스펙 변경 시 RAG 쿼리 생성 로직도 함께 수정 필요  
**MVP 기준**: 추후 항목 추가/삭제 가능하도록 Pydantic optional 필드로 관리

---

### ADR-007: 임베딩 공유 레이어 — src/embedding/ 분리

**결정**: `pipeline/embedder.py` 대신 `src/embedding/`을 독립 공유 레이어로 분리. `EmbedderProtocol`(typing.Protocol)과 구체 구현체(KoSimCSE)를 두고 `pipeline/`과 `retriever/` 모두 임포트  
**이유**: pipeline(인덱싱 시)과 retriever(쿼리 임베딩 시) 양쪽에서 동일 모델이 필요한데, 한 레이어가 소유하면 다른 레이어가 잘못된 방향으로 임포트해야 함. 공유 인프라 레이어로 분리하면 순환 의존 없이 해결  
**레이어 의존 방향**: `embedding/`은 다른 `src/` 모듈을 임포트하지 않음 (최하위 인프라)  
**트레이드오프**: ARCHITECTURE.md의 기존 디렉토리 구조 변경 필요

---

### ADR-008: ChromaDB 클라이언트 — 동기 클라이언트 + asyncio.to_thread 래핑

**결정**: chromadb 1.5.7에서 `AsyncEphemeralClient` / `AsyncPersistentClient`가 존재하지 않음을 실제 설치 환경에서 확인. 로컬 async 클라이언트는 지원되지 않고 `AsyncHttpClient`(원격 서버 전용)만 제공됨. 따라서 동기 클라이언트(`EphemeralClient`, `PersistentClient`)를 사용하고, 호출 지점에서 `asyncio.to_thread()`로 래핑한다.

```python
import asyncio, chromadb

# 클라이언트 생성
client = await asyncio.to_thread(chromadb.EphemeralClient)           # 테스트/로컬
client = await asyncio.to_thread(chromadb.PersistentClient, path)    # 프로덕션

# 컬렉션 연산
col = await asyncio.to_thread(client.get_or_create_collection, name, metadata={"hnsw:space": "cosine"})
```

**이유**: 동기 클라이언트는 in-process HNSW 연산이므로 네트워크 I/O가 없음. `asyncio.to_thread()`로 스레드풀에 위임하면 이벤트 루프를 블로킹하지 않으면서 FastAPI async route handler와 통합 가능  
**트레이드오프**: 모든 ChromaDB 호출에 `asyncio.to_thread()` 래핑 필요. 코드가 다소 장황해짐. 그러나 HTTP 서버를 별도 운영하는 것보다 MVP 복잡도가 훨씬 낮음

---

> ADR을 추가할 때마다 번호를 순차 증가시킨다. 한번 기록된 결정은 변경하지 않고, 번복 시 새 ADR을 추가한다.

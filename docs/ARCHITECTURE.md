# 아키텍처: BokjiDream RAG

## 디렉토리 구조

```
src/
├── api/                   # FastAPI 앱 + 라우터
│   ├── main.py            # 앱 초기화, lifespan (KoSimCSEEmbedder 1회 생성)
│   ├── deps.py            # 공통 의존성 (get_embedder — 순환 임포트 방지)
│   └── routes/
│       └── welfare.py     # POST /welfare/search, GET /welfare/{serv_id}
├── crawler/               # 데이터 수집 레이어
│   ├── client.py          # HTTP 클라이언트 (httpx)
│   ├── collect.py         # 수집 진입점 (배치 실행용)
│   ├── welfare_list.py    # 공공데이터 목록 조회 API 래퍼
│   └── welfare_detail.py  # 공공데이터 상세 조회 API 래퍼
│   # 추후 확장: welfare_crawler.py (복지로 Playwright), pdf_downloader.py (HWP/PDF)
├── embedding/             # 공유 임베딩 인프라 (다른 src/ 모듈 임포트 없음)
│   ├── protocol.py        # EmbedderProtocol (typing.Protocol)
│   └── kosimcse.py        # sentence-transformers 구체 구현체
├── pipeline/              # 전처리 + 임베딩 + 인덱싱
│   ├── chunker.py         # 문서 청크 분할
│   └── index.py           # ChromaDB 인덱싱 진입점 (embedding/ 임포트)
├── retriever/             # 검색 로직
│   └── search.py          # 쿼리 임베딩 + ChromaDB 유사도 검색
├── db/                    # DB 클라이언트 초기화
│   └── chroma.py          # ChromaDB 싱글턴 클라이언트
├── models/                # 도메인 모델 + Pydantic 스키마
│   └── welfare.py         # WelfareRaw, SearchRequest, SearchResult, SearchResponse, WelfareDetail
└── utils/                 # 공통 유틸
    └── pdf_parser.py      # PDF 텍스트 추출

tests/
├── unit/
│   ├── test_models.py     # step 1
│   ├── test_db.py         # step 2
│   ├── test_embedder.py   # step 3
│   ├── test_chunker.py    # step 5
│   ├── test_index.py      # step 5
│   └── test_search.py     # step 6
└── integration/
    ├── test_crawler.py    # step 4
    └── test_api.py        # step 7

data/
├── raw/                   # 수집 원본 (PDF, JSON)
└── processed/             # 청크 분할 후 데이터
```

## 레이어 구조 및 의존 방향

```
[외부 데이터 소스]
  공공데이터 API / 복지로 / PDF
        ↓
[crawler/]  ← 수집만. DB 접근 없음
        ↓
[pipeline/] ← 전처리 + 인덱싱 (embedding/ 임포트)
        ↓
[db/chroma.py] ← 동기 클라이언트 + asyncio.to_thread 싱글턴 (ADR-008)
        ↑
[retriever/] ← 쿼리 생성 + ChromaDB 검색 (embedding/ 임포트)
        ↑
[api/] ← FastAPI. 외부 인터페이스
        ↑
[LangGraph 오케스트레이터] ← 유일한 클라이언트

[embedding/] ← 공유 인프라. pipeline/과 retriever/ 양쪽에서 임포트. 다른 src/ 모듈 의존 없음
```

**레이어 간 직접 임포트 금지**: `api → retriever → db` 방향만 허용. `crawler`가 `retriever`를 임포트하는 등 역방향 절대 금지.

## 핵심 패턴

- **Repository 패턴 생략**: ChromaDB 클라이언트를 `db/chroma.py`에서 직접 노출. 추상화 레이어보다 단순함 우선
- **Pydantic 스키마**: 모든 API 입출력은 `models/welfare.py`의 Pydantic 모델 사용
- **싱글턴 ChromaDB**: 앱 시작 시 1회 초기화, DI로 주입

## 데이터 흐름

### 인덱싱 (배치)
```
복지로 크롤링 / 공공API / PDF
  → crawler/collect.py
  → pipeline/chunker.py   (청크 분할, 1000자 / 200자 오버랩)
  → embedding/kosimcse.py  (한국어 임베딩, pipeline/index.py에 DI로 주입)
  → pipeline/index.py     (ChromaDB upsert)
```

### 검색 (실시간)
```
LangGraph POST /welfare/search { 유저 조건 }
  → api/routes/welfare.py
  → retriever/search.py   (쿼리 생성 + 임베딩 + ChromaDB similarity_search)
  → List[SearchResult] JSON 반환

LangGraph GET /welfare/{serv_id}
  → api/routes/welfare.py
  → retriever/search.py   (ChromaDB ID 조회)
  → WelfareDetail JSON 반환
```

## 외부 의존성

| 서비스 | 용도 | 환경변수 |
|--------|------|----------|
| 공공데이터포털 API | 복지 서비스 목록 수집 | `PUBLIC_DATA_API_KEY` |
| AWS S3 | PDF 원본 저장 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| AWS Lambda | 배치 크롤링 스케줄 | (인프라 팀 담당) |

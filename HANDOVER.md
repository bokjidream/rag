# HANDOVER — bokjidream-rag

> 복지서비스 RAG 파이프라인. 공공 복지 서비스 데이터를 크롤링/청킹/임베딩해서 ChromaDB에 저장하고, 검색 API를 LangGraph 팀에 제공하는 백엔드.
> 2026-07-28 기준 스냅샷입니다.

## 1. 한눈에 보기

- **역할**: 사용자 프로필(나이/소득/거주 등)을 입력받아 적합한 복지 서비스를 검색해 JSON으로 반환. 실제 대화형 응답 생성은 LangGraph 팀이 담당하고, 이 프로젝트는 순수 검색/랭킹 백엔드.
- **핵심 스택**: FastAPI + ChromaDB(로컬 persistent) + `jhgan/ko-sroberta-multitask`(sentence-transformers, 로컬 임베딩, 외부 API 비용 없음) + Playwright/httpx 크롤러.
- **왜 이렇게 만들어졌는지**: `docs/ADR.md`에 9개 결정이 기록되어 있음 (ChromaDB 선택 이유, 임베딩 모델 고정 이유, FastAPI 선택 이유, 크롤링 전략, 검색 API가 자연어가 아닌 구조화 프로필을 받는 이유 등).
- **GitHub 이슈**: 현재 열린 이슈 0건, #2~#28 전부 closed. 명시적인 backlog는 없고, 아래 6절 "남아있는 작업"이 사실상의 backlog에 해당함.

## 2. 데이터 흐름 (아키텍처)

```
crawler/  →  pipeline/ (청킹)  →  embedding/ (벡터화)  →  db/ (ChromaDB)
                                                              ↑
                                                         retriever/ (검색·재순위·자격판정)
                                                              ↑
                                                          api/ (FastAPI)
                                                              ↑
                                                        LangGraph 팀
```

- **`src/crawler/`**: 공공데이터포털 API(목록/상세) + Playwright로 복지로 상세페이지 크롤링. `PUBLIC_DATA_API_KEY`가 필요하고, 사용자 인증이 필요한 개인화 페이지·파일은 대리 다운로드하지 않음(ADR-009).
- **`src/pipeline/chunker.py`**: 청킹 전략이 두 가지 공존함 — ① 레거시 `chunk_item()`(문서 전체 슬라이딩 윈도우, baseline `data/chroma`가 사용) ② `chunk_metadata_sections()`(필드별 section-aware 청킹, `data/chroma-section-aware`가 사용).
- **`src/embedding/kosimcse.py`**: 임베딩 모델은 하드코딩되어 있음(`jhgan/ko-sroberta-multitask`, ADR-002). 모델을 바꾸면 전체 재인덱싱이 필요함.
- **`src/db/chroma.py`**: 컬렉션은 `hnsw:space=cosine`으로 고정되어 있고, retriever의 `score = max(0, 1 - distance)` 계산이 이 값에 의존함.
- **`src/retriever/`**: `search.py`(메인 검색 오케스트레이션) → `intent.py`(프로필→랭킹용 의도 정규화) → `rerank.py`(섹션 가중치 + 네거티브 조건 페널티 + 테마 조정) → `eligibility.py`(패턴 매칭 기반 자격 판정 가드레일, LLM 미사용). `unlikely` 판정된 후보는 자동으로 응답에서 제외됨.
- **`src/api/`**: `POST /welfare/search`, `GET /welfare/{serv_id}` 2개 엔드포인트. 자세한 스키마는 7절 참고.

## 3. 실행 방법

**로컬**
```bash
make setup              # venv 생성 + pip install -e ".[dev]"
uvicorn src.api.main:app --reload   # API 서버 (Makefile엔 없음, 직접 실행)
```

**배치 인덱싱 순서**
```bash
PUBLIC_DATA_API_KEY=<key> python scripts/run_pipeline.py   # 크롤링 + baseline 인덱싱 → data/chroma
python scripts/build_section_aware_index.py                # baseline을 읽어 section-aware 인덱스 생성 → data/chroma-section-aware
python scripts/evaluate_search.py                           # 품질 검증 (100케이스 평가셋)
```

**Docker** (`Dockerfile`, `docker-compose.yml`, 자세한 절차는 `docs/docker.md`)
- docker-compose 기본값은 baseline이 아니라 `welfare_services_section_aware` 컬렉션(`data/chroma-section-aware` 마운트)을 사용함. 이 DB는 git에 없고 로컬에서 `build_section_aware_index.py`로 직접 생성해야 함.
- 스모크테스트는 원본 DB를 직접 마운트하지 않고 `mktemp -d`로 뜬 복사본을 사용함 — Chroma가 SQLite 런타임 메타데이터를 변경하기 때문(`docs/docker.md`).

**환경변수** (`.env.example` 기준, 실제 값은 `.env`에만 존재 — 공유 금지)
| 변수 | 용도 |
|---|---|
| `PUBLIC_DATA_API_KEY` | 공공데이터포털 발급 키 |
| `CHROMA_PERSIST_DIR` | 기본 `data/chroma` |
| `CHROMA_MODE` | `persistent`(기본) \| `ephemeral` |
| `API_HOST`, `RAG_API_PORT` | 기본 포트 `8002` |
| `WELFARE_COLLECTION_NAME`, `WELFARE_ADAPTIVE_FETCH`, `CHROMA_HOST_DIR`, `API_PORT` | docker-compose 전용 |

## 4. 테스트

```bash
make test-unit           # tests/unit (커버리지 포함)
make test-integration    # API e2e + 크롤러 실제 호출
make test-search-quality # tests/integration/test_search_quality.py (합성 회귀)
make test-all
make lint                # ruff
make typecheck           # mypy --strict
```
`evaluate_search.py`는 위 테스트와 별개로 존재하는 100케이스 품질 평가 하네스(`must`/`acceptable`/`conditional`/`excluded`/`ambiguous` 계약 기반 recall/MRR 계산).

## 5. 데이터 상태 — ChromaDB가 GitHub에 올라가 있는가?

**결론: 벡터 DB(`data/chroma/`)만 커밋되어 있고, 원본/중간 데이터는 올라가 있지 않음.**

| 경로 | Git 상태 | 내용 |
|---|---|---|
| `data/raw/*` | gitignore (`.gitkeep`만 추적) | 크롤링 원본 |
| `data/processed/*` | gitignore (`.gitkeep`만 추적) | 청킹 전 중간 데이터, `application_forms_fetch_state.json`(신청서식 수집 진행상태 캐시) 등 |
| `data/chroma/` | **커밋됨** (`chroma.sqlite3` 34MB + HNSW 인덱스 파일) | baseline 벡터 DB, 컬렉션 `welfare_services`, 570 chunks / 413 services |
| `data/chroma-section-aware/` | gitignore, 로컬 전용 | section-aware 재인덱싱 결과, 2,424 chunks / 413 services. **Docker 배포 기본값이 이걸 참조하는데 git엔 없음** |

즉 "청킹해서 정리한 텍스트"가 올라간 게 아니라, 그 청크를 임베딩까지 마친 완성된 벡터 인덱스만 seed용으로 커밋되어 있음(`e2fc5f5 chore: ChromaDB 데이터 포함`, `.gitignore` 주석에도 "seed DB 파일 자체는 git에 포함"이라 명시). 새로 클론했을 때 바로 baseline 검색 데모가 동작하게 하려는 의도로 보임.

참고로 `git status`상 `data/chroma/chroma.sqlite3`가 현재 modified 상태인데, Chroma 런타임이 열릴 때 SQLite 메타데이터를 건드리는 특성 때문일 수 있음(README에 명시된 알려진 동작).

## 6. 최근 작업 이력 & 남아있는 작업

**최근 커밋 2건**
- `f931fe4` (#26) — 자격판정 가드레일, 100케이스 평가셋 확장, section-aware chunking, 재순위/테마 조정, 평가 계약 정비
- `27fa814` (#25) — Docker 기반 배포 준비, `ApiConfig`/`resolve_api_config()`, `validate_existing_collection()`

**`docs/goal.md` ~ `goal7.md`** (git엔 아직 없는 상태): 향후 계획이 아니라 각 개발 단계에서 실제로 내려진 작업 지시문 원본. `goal.md`(자격판정 가드레일) → `goal2.md`(평가셋 100개 확장) → `goal3.md`(section-aware chunking) → `goal4.md`(네거티브 페널티+재순위 2차) → `goal5.md`(평가 계약 재분류) → `goal6.md`(#26 최종 구현) → `goal7.md`(#25 Docker 배포) 순서로, "왜 지금 이렇게 되어 있는지"의 근거 자료에 해당함.

**남아있는 것으로 보이는 작업 (이슈로 등록되진 않음)**
1. 검색 품질은 아직 미완결 상태 — `docs/search-quality-plan.md`에 잔여 실패 유형이 나열되어 있음(아동/보육 vs 한부모·자립준비청년 오탐, 저소득 테마 구분 미흡, 61/64/65세 경계 처리, 장애인 비소득/비주거 의도 섹션 가중치 등). baseline recall@5 0.68 수준. 단, `search-quality-plan.md`가 `goal6.md`(#26) 시점 이후 최신 상태까지 반영됐는지는 불확실함 — 코드(`rerank.py`의 `intent_theme_adjustment` 등)는 구현되어 있는데 문서가 그 이전 단계 서술일 가능성이 있음.
2. Docker 기본 설정이 참조하는 `data/chroma-section-aware`를 프로덕션 환경에 어떻게 옮길지에 대한 배포 파이프라인이 아직 없음.
3. 미구현 확장 기능: `basfrmList`(신청서식) HWP/PDF 파싱(ADR-005에서 예정됐으나 미착수), AWS Lambda 배치 갱신(PRD/ARCHITECTURE엔 "인프라 팀 담당"으로만 언급, 실제 미구현).

**문서와 코드가 어긋나는 부분** (읽을 때 헷갈리기 쉬운 지점)
- `docs/ARCHITECTURE.md`는 초기(0-mvp) 시점 문서라 이후 추가된 `retriever/intent.py`, `retriever/eligibility.py`, `retriever/rerank.py`, `api/config.py`, `api/deps.py` 등을 언급하지 않음.
- `ARCHITECTURE.md`가 언급하는 `utils/pdf_parser.py`, `crawler/welfare_crawler.py`, `crawler/pdf_downloader.py`는 실제로 존재하지 않음.
- `docs/PRD.md`의 `SearchResult` 스펙에 없는 `eligibility_status`/`eligibility_reasons`/`missing_fields`/`evidence` 필드가 실제 코드(`src/models/welfare.py`)엔 추가되어 있음.

## 7. LangGraph 팀 인터페이스

**호출 흐름**: `POST /welfare/search`로 top-k 검색 → 사용자가 고른 `serv_id`로 `GET /welfare/{serv_id}` 상세 조회. `serv_id`는 검색 응답에서만 얻는다고 가정(직접 호출 불가).

**요청 — `SearchRequest`** (`src/models/welfare.py`)
```python
age: int                                                   # 필수, 0~130
income_level: Literal["기초생활수급자","차상위계층","저소득","일반"]  # 필수
household_size: int | None = None                          # 1~20
marital_status: Literal["미혼","기혼","이혼","사별"] | None = None
has_children: bool | None = None
disability: bool = False
disability_severity: Literal["경증","중증"] | None = None   # disability=True일 때만 허용
employment_status: Literal["취업","실업","비경제활동"] | None = None
pregnant: bool = False                                     # PRD엔 없지만 코드엔 존재
region: str | None = None                                  # 메타데이터 필터 아님, soft 텍스트 매칭
top_k: int = 5                                             # 1~50
```
`disability`와 `disability_severity` 조합이 안 맞으면 pydantic validator가 예외를 던짐.

**응답 — `SearchResult`** (PRD 대비 자격판정 필드가 추가되어 있음)
```python
serv_id, serv_nm, serv_dgst, department: str
score: float
trgter_indvdl, intrs_thema: list[str]
eligibility_status: Literal["likely","needs_more_info","unlikely"] = "likely"  # unlikely는 응답에서 자동 제외됨
eligibility_reasons: list[str]
missing_fields: list[str]
evidence: list[{field: str, text: str}]
```

**상세 조회 — `WelfareDetail`**: `serv_id, serv_nm, serv_dgst, tgtr_dtl_cn, slct_crit_cn, alw_serv_cn, sprt_cyc_nm, srv_pvsn_nm, trgter_indvdl, intrs_thema, application_url, application_method, application_forms: [{title,url,file_type}], required_documents: list[str]`. 없는 `serv_id`는 404.

**사전 협의 필요 항목** (`docs/PRD.md`): `application_fields`는 제거 예정이나 협의 전까지 유지, `eligibility_reason`은 이 서비스가 아니라 LangGraph LLM이 생성, 응답 구조 변경 시 반드시 사전 협의.

## 8. 참고 문서 지도

| 문서 | 용도 |
|---|---|
| `docs/ADR.md` | 왜 이렇게 만들었는지 (9개 결정) — **가장 먼저 읽을 것** |
| `docs/PRD.md` | LangGraph 팀과 합의된 API 스펙, MVP 범위 |
| `docs/ARCHITECTURE.md` | 레이어 의존 방향 (단, 최신 코드와 일부 어긋남, 위 6절 참고) |
| `docs/docker.md` | Docker 배포/스모크테스트 절차 |
| `docs/search-quality-plan.md` | 검색 품질 개선 이력, 잔여 실패 유형 |
| `docs/goal.md` ~ `goal7.md` | (미커밋) 각 단계별 실제 작업 지시 원본, 근거 자료용 |

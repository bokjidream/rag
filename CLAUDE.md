# 프로젝트: BokjiDream RAG

## 기술 스택

| 항목 | 선택 |
|------|------|
| 런타임/프레임워크 | Python 3.9+ / FastAPI |
| 언어 | Python 3.9 (type hints, strict mypy) |
| 벡터 DB | ChromaDB |
| 임베딩 | sentence-transformers (한국어 모델) |
| RAG 프레임워크 | LangChain |
| 크롤링 | Playwright + httpx |
| 테스트 | pytest + pytest-asyncio |
| 린트/포매터 | ruff |
| 타입 체커 | mypy (strict) |
| 패키지 매니저 | pip (pyproject.toml) |

## 명령어

| 용도 | 커맨드 |
|------|--------|
| 개발 서버 | `uvicorn src.api.main:app --reload` |
| 테스트 | `pytest --cov=src --cov-report=term-missing` |
| 린트 | `ruff check src/ tests/` |
| 타입 체크 | `mypy src/` |

## 아키텍처 규칙

- CRITICAL: RAG 검색 결과는 반드시 합의된 `List[SearchResult]` JSON 스키마로 반환한다 — LangGraph 팀과 인터페이스를 임의로 변경하지 말 것
- CRITICAL: 사용자 개인정보(나이, 소득, 가구원 수 등)를 외부 API나 로그에 전송하지 말 것 (Sovereign AI 요건)
- CRITICAL: 레이어 경계를 지킨다 — 크롤러는 `src/crawler/`에만, 임베딩/인덱싱은 `src/pipeline/`에만, 검색은 `src/retriever/`에만
- ChromaDB 클라이언트는 `src/db/` 에서만 초기화하고 싱글턴으로 관리한다
- 외부 API(공공데이터포털, 복지로) 호출은 `src/crawler/` 레이어에서만 수행한다

## 개발 프로세스

- CRITICAL: 새 기능 구현 시 반드시 테스트를 먼저 작성하고 (RED), 테스트가 통과하는 최소 구현을 작성할 것 (GREEN → REFACTOR)
- 커밋 메시지는 conventional commits 형식을 따를 것 (feat:, fix:, docs:, refactor:)
- 테스트 커버리지 80% 이상 유지

## 하네스 실행 규칙

- CRITICAL: 각 step을 시작하기 전에 해당 step의 `.md` 파일과 `docs/` 전체를 반드시 읽을 것
- CRITICAL: step 완료 조건은 `## Acceptance Criteria` 커맨드가 실제로 통과하는 것이다 — 단순히 코드 작성으로 완료가 아님
- CRITICAL: AC 커맨드 실패 시 즉시 원인 분석 후 수정하고 재실행한다 (최대 3회 재시도, 초과 시 `"status": "error"`)
- 사용자 개입이 필요한 상황(API 키, 외부 서비스 등)이 되면 즉시 `"status": "blocked"`로 마킹하고 중단한다

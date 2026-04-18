# bokjidream-rag

## 역할

복지서비스 RAG 파이프라인 서비스.
LangGraph 팀에 검색 결과를 JSON으로 반환하는 것이 핵심 책임이다.

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| API 서버 | FastAPI |
| 크롤링 | Playwright |
| RAG 파이프라인 | LangChain |
| 벡터 DB | ChromaDB |
| 언어 | Python |

## 폴더 구조

```
data/
├── raw/          # 크롤링 원본 데이터
└── processed/    # 청킹·임베딩 전처리 결과

src/
├── crawler/      # Playwright 기반 복지서비스 크롤러
├── pipeline/     # LangChain RAG 파이프라인 (청킹, 임베딩, 인덱싱)
└── retriever/    # ChromaDB 검색, FastAPI 엔드포인트
```

## API 계약 (LangGraph 팀과의 인터페이스)

검색 결과는 반드시 아래 JSON 형식으로 반환한다.

```json
{
  "query": "string",
  "results": [
    {
      "id": "string",
      "content": "string",
      "metadata": {
        "source": "string",
        "service_name": "string",
        "url": "string"
      },
      "score": 0.0
    }
  ],
  "total": 0
}
```

- 필드 추가는 가능하나 기존 필드 삭제·변경은 LangGraph 팀과 협의 후 진행한다.
- 오류 응답도 일관된 형식을 유지한다: `{ "error": "string", "detail": "string" }`

## 개발 규칙

### Python 스타일

- 타입 힌트 필수 (모든 함수 파라미터·반환값)
- 비동기: `async/await` 사용 (FastAPI + Playwright 모두 비동기)
- 포매터: `ruff` (format + lint 통합)
- 타입 체커: `mypy`

### 에러 처리

- 크롤러: 실패한 URL은 로그에 기록하고 건너뜀 (파이프라인 중단 금지)
- 파이프라인: 임베딩 실패 시 해당 청크만 skip, 나머지 계속 진행
- API: FastAPI exception handler로 일관된 에러 응답 보장

### 데이터 흐름

```
Playwright 크롤링
  → data/raw/ 저장
  → LangChain 청킹·임베딩
  → data/processed/ 저장
  → ChromaDB 인덱싱
  → FastAPI /search 엔드포인트
  → LangGraph 팀
```

### 테스트

- `src/crawler/` : Playwright 크롤러는 실제 요청 대신 fixture HTML로 단위 테스트
- `src/pipeline/` : 청킹·임베딩 로직은 샘플 문서로 단위 테스트
- `src/retriever/` : FastAPI TestClient + ChromaDB in-memory 컬렉션으로 통합 테스트
- 커버리지 80% 이상 유지

## Sprint Contract

작업 시작 전 완료 기준을 합의한다:
- **완료 기준**: 테스트 통과 + 커버리지 80% + ruff/mypy 오류 없음
- 모호한 요청은 먼저 질문해서 명확히 한 뒤 시작

## 참고 문서 (docs/)

- 코드 작성 시 → [docs/conventions.md](docs/conventions.md)
- API 응답 설계 시 → [docs/api-contract.md](docs/api-contract.md)
- 아키텍처 파악 시 → [docs/architecture.md](docs/architecture.md)

## 경계 (하면 안 되는 것)

- `.env` 파일 직접 수정 금지 (`.env.example`만 수정)
- `data/raw/`, `data/processed/` 수동 삭제 금지
- LangGraph API 계약 필드 변경 시 반드시 사용자 확인

## 적용 Rules

- `.claude/rules/` — 이 프로젝트 전용 규칙 (glob 패턴으로 조건부 로드)
- `~/.claude/rules/common/` — 공통 원칙 (코딩 스타일, TDD, 보안 등)
- `~/.claude/rules/python/` — Python 특화 규칙 (ruff, mypy, pytest)

> web rules는 이 프로젝트에 적용하지 않는다.

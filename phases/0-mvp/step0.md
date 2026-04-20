# Step 0: project-setup

## 읽어야 할 파일

먼저 아래 파일들을 읽고 프로젝트의 아키텍처와 설계 의도를 파악하라:

- `docs/ARCHITECTURE.md`
- `docs/ADR.md`
- `pyproject.toml`
- `.env.example`

## 작업

`pyproject.toml`과 `.env.example`은 이미 완성되어 있다. 이 step의 목적은
**디렉토리 구조와 빈 진입점 파일만** 만드는 것이다.

### 1. 누락된 src/ 디렉토리 생성

아래 디렉토리와 `__init__.py`를 생성한다:

```
src/
├── api/
│   ├── __init__.py
│   ├── deps.py          (빈 파일 — step 7에서 구현)
│   └── routes/
│       └── __init__.py
├── db/
│   └── __init__.py
├── embedding/
│   └── __init__.py
├── models/
│   └── __init__.py
└── utils/
    └── __init__.py
```

`src/crawler/`, `src/pipeline/`, `src/retriever/`의 `__init__.py`는 이미 존재한다.

### 2. tests/ 디렉토리 재구성

현재 `tests/crawler/`, `tests/pipeline/`, `tests/retriever/`를 제거하고
아래 구조로 새로 만든다:

```
tests/
├── __init__.py       (이미 존재)
├── unit/
│   └── __init__.py
└── integration/
    └── __init__.py
```

### 3. data/chroma 디렉토리 생성

```
data/
├── raw/          (.gitkeep 이미 존재)
├── processed/    (.gitkeep 이미 존재)
└── chroma/
    └── .gitkeep
```

### 4. .gitignore 확인

아래 항목이 `.gitignore`에 포함되어 있는지 확인하고, 없으면 추가한다:

```
data/chroma/
.env
.coverage
```

## Acceptance Criteria

```bash
ruff check src/      # 린트 오류 없음
mypy src/            # 타입 오류 없음 (빈 __init__.py만 있으므로 통과해야 함)
python -c "from pathlib import Path; dirs=['tests/unit','tests/integration','src/api/routes','src/db','src/embedding','src/models','src/utils','data/chroma']; missing=[d for d in dirs if not Path(d).is_dir()]; assert not missing, f'누락된 디렉토리: {missing}'"
```

## 검증 절차

1. 위 AC 커맨드를 순서대로 실행한다.
2. 아키텍처 체크리스트:
   - `ARCHITECTURE.md`의 디렉토리 구조와 일치하는가?
   - `data/chroma/`가 `.gitignore`에 포함되어 있는가?
   - `tests/unit/`, `tests/integration/`이 생성되었는가?
3. 결과에 따라 `phases/0-mvp/index.json`의 step 0을 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "src/ 전체 디렉토리 구조, tests/unit+integration 생성 완료"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- 비즈니스 로직을 구현하지 마라. 빈 `__init__.py`와 디렉토리 생성만 한다.
- `pyproject.toml` 의존성을 임의로 추가하거나 변경하지 마라.
- `.env` 파일에 실제 값을 넣지 마라.

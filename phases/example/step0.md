# Step 0: project-setup

## 읽어야 할 파일

먼저 아래 파일들을 읽고 프로젝트의 아키텍처와 설계 의도를 파악하라:

- `/docs/ARCHITECTURE.md`
- `/docs/ADR.md`

## 작업

아래 디렉토리 구조를 생성하고 각 계층의 진입점 파일(빈 index 파일)을 만든다.

```
src/
├── app/
│   └── api/
├── components/
├── types/
│   └── index.ts      ← 공통 타입 export 진입점 (비워둠)
├── lib/
│   └── utils.ts      ← 공통 유틸 진입점 (비워둠)
└── services/
```

그리고 아래 환경변수 파일을 생성한다:

- `.env.example` — 필요한 환경변수 키 목록 (값은 비워둠, 커밋 가능)
- `.env.local` — 실제 값 입력용 (`.gitignore`에 포함되어 있음)

`.env.example`에 들어갈 키 목록:
```
DATABASE_URL=
```

## Acceptance Criteria

```bash
pnpm build   # 컴파일 에러 없음
pnpm test    # 테스트 통과 (아직 테스트 없으면 "no tests found" 도 OK)
```

## 검증 절차

1. 위 AC 커맨드를 실행한다.
2. 아키텍처 체크리스트:
   - `ARCHITECTURE.md`의 디렉토리 구조를 따르는가?
   - `.env.local`이 `.gitignore`에 포함되어 있는가?
3. 결과에 따라 `phases/example/index.json`의 step 0을 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "프로젝트 기본 디렉토리 구조 및 .env.example 생성 완료"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- 비즈니스 로직을 구현하지 마라. 이 step은 디렉토리와 빈 진입점 파일 생성만 한다.
- `.env.local`에 실제 시크릿 값을 넣지 마라.

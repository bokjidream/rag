# Step 1: core-types

## 읽어야 할 파일

먼저 아래 파일들을 읽고 프로젝트의 아키텍처와 설계 의도를 파악하라:

- `/docs/ARCHITECTURE.md`
- `/docs/ADR.md`
- `src/types/index.ts`   ← Step 0에서 생성된 파일

이전 step에서 만들어진 코드를 꼼꼼히 읽고, 설계 의도를 이해한 뒤 작업하라.

## 작업

`src/types/` 아래에 아래 타입 정의를 작성한다.

### `src/types/user.ts`

```ts
export interface User {
  id: string
  email: string
  name: string
  createdAt: Date
}

export interface CreateUserInput {
  email: string
  name: string
}
```

### `src/types/index.ts`

모든 타입을 re-export한다:

```ts
export * from './user'
// 이후 추가되는 타입 파일도 여기서 export
```

타입 설계 원칙:
- `id`는 항상 `string` (UUID 또는 외부 서비스 ID)
- 뮤테이션 입력 타입(`CreateXxxInput`, `UpdateXxxInput`)을 도메인 타입과 분리
- `Date` 타입은 서버 경계 통과 시 직렬화 고려 필요 (ISO string)

## Acceptance Criteria

```bash
pnpm build   # 타입 에러 없음
```

## 검증 절차

1. 위 AC 커맨드를 실행한다.
2. 아키텍처 체크리스트:
   - 타입 정의가 `src/types/`에만 위치하는가?
   - `src/types/index.ts`에서 모두 re-export하는가?
3. 결과에 따라 `phases/example/index.json`의 step 1을 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "User, CreateUserInput 타입 정의 완료. src/types/index.ts에서 re-export."`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- 비즈니스 로직이나 API 핸들러를 이 step에서 구현하지 마라.
- `any` 타입을 사용하지 마라. 타입 미확정 시 `unknown`을 쓰고 주석으로 이유를 남겨라.

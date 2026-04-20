# Step 2: api-layer

## 읽어야 할 파일

먼저 아래 파일들을 읽고 프로젝트의 아키텍처와 설계 의도를 파악하라:

- `/docs/ARCHITECTURE.md`
- `/docs/ADR.md`
- `src/types/user.ts`         ← Step 1에서 정의된 타입
- `src/types/index.ts`

이전 step에서 만들어진 코드를 꼼꼼히 읽고, 설계 의도를 이해한 뒤 작업하라.

## 작업

Next.js App Router 기준으로 User CRUD API 라우트를 구현한다.

### `src/app/api/users/route.ts`

```ts
// GET /api/users   — 전체 목록
// POST /api/users  — 생성

import { type NextRequest, NextResponse } from 'next/server'
import { type CreateUserInput } from '@/types'

export async function GET(req: NextRequest): Promise<NextResponse> { ... }
export async function POST(req: NextRequest): Promise<NextResponse> { ... }
```

### `src/app/api/users/[id]/route.ts`

```ts
// GET    /api/users/:id  — 단건 조회
// PATCH  /api/users/:id  — 수정
// DELETE /api/users/:id  — 삭제

import { type NextRequest, NextResponse } from 'next/server'

export async function GET(req: NextRequest, { params }: { params: { id: string } }): Promise<NextResponse> { ... }
export async function PATCH(...): Promise<NextResponse> { ... }
export async function DELETE(...): Promise<NextResponse> { ... }
```

구현 지침:
- DB 연동 없이 인메모리 배열(`let users: User[] = []`)로 구현한다 (추후 Repository로 교체 예정)
- 에러 응답은 `{ error: string }` 형태로 일관되게 반환
- 존재하지 않는 ID 요청 시 404 반환

### `src/app/api/users/route.test.ts`

GET, POST 각각에 대한 단위 테스트를 작성한다:

```ts
describe('GET /api/users', () => {
  it('빈 배열을 반환한다', async () => { ... })
})

describe('POST /api/users', () => {
  it('유효한 입력으로 사용자를 생성한다', async () => { ... })
  it('email이 없으면 400을 반환한다', async () => { ... })
})
```

## Acceptance Criteria

```bash
pnpm build   # 컴파일 에러 없음
pnpm test    # 새로 작성한 API 테스트 통과
```

## 검증 절차

1. 위 AC 커맨드를 실행한다.
2. 아키텍처 체크리스트:
   - API 로직이 `app/api/` 경로에만 위치하는가? (CRITICAL 규칙)
   - 테스트가 작성되었는가?
3. 결과에 따라 `phases/example/index.json`의 step 2를 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "User CRUD API (GET/POST/PATCH/DELETE) 인메모리 구현 + 테스트 완료"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- DB 연결 코드를 이 step에서 작성하지 마라. 인메모리 구현으로 충분하다.
- 클라이언트 컴포넌트(`'use client'`)에서 이 API를 직접 호출하는 코드를 만들지 마라. (CRITICAL 규칙 위반)

# 코드 스타일 — 항상 적용

## 네이밍

- 변수·함수·모듈: `snake_case`
- 클래스: `PascalCase`
- 상수: `UPPER_SNAKE_CASE`

## 타입 힌트

- 모든 함수의 파라미터와 반환값에 타입 힌트 필수
- `Any` 사용 시 `from typing import Any` 명시 import

## 함수·파일 크기

- 함수: 50줄 이내
- 파일: 400줄 이내 (초과 시 모듈 분리)

## async/await

- FastAPI 엔드포인트와 Playwright는 모두 `async def`
- `sync_playwright` 사용 금지
- CPU-bound 작업은 `asyncio.run_in_executor` 활용

## 로깅

- `print()` 금지 → `logging.getLogger(__name__)` 사용
- 로그에 민감 정보(개인정보, 토큰) 포함 금지

## 참고

코드 작성 시 → [docs/conventions.md](../../docs/conventions.md)

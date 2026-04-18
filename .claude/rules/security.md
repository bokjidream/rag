---
glob: "**/*.py"
---

# 보안 규칙 — 모든 .py 파일 작업 시 적용

## 환경변수

- `.env` 파일 직접 수정 금지 → `.env.example`만 수정
- 환경변수는 `os.getenv()` 또는 `pydantic BaseSettings` 사용
- 시크릿·API 키 소스코드 하드코딩 절대 금지

## 외부 요청

- 크롤러 URL은 `BOKJIRO_BASE_URL` 기반으로만 구성
- 임의의 외부 URL 하드코딩 금지

## 입력 검증

- 사용자 입력은 Pydantic 모델로 검증 후 사용
- ChromaDB 쿼리에 user input 직접 삽입 금지 (파라미터화 사용)

## 로그

- 에러 메시지에 내부 경로, 스택 트레이스를 사용자에게 노출 금지
- 개인정보, 검색어 원문은 운영 로그에 최소화

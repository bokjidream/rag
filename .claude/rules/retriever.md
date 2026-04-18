---
glob: "src/retriever/**"
---

# 리트리버 규칙 — src/retriever/ 작업 시 적용

## API 응답 스키마

검색 응답은 반드시 아래 형식을 따른다 (→ [docs/api-contract.md](../../docs/api-contract.md)):

```json
{
  "query": "string",
  "results": [{"id", "content", "metadata", "score"}],
  "total": 0
}
```

- 기존 필드(`query`, `results`, `total`, `id`, `content`, `metadata`, `score`) **삭제·변경 금지**
- 필드 추가 전 [docs/api-contract.md](../../docs/api-contract.md) 확인

## 에러 응답

FastAPI exception handler로 모든 에러를 일관된 형식으로 반환:

```json
{"error": "ErrorType", "detail": "설명"}
```

## FastAPI 설계

- 엔드포인트 함수는 모두 `async def`
- 입력 검증은 Pydantic 모델로 처리
- 의존성 주입(Depends)으로 ChromaDB 클라이언트 관리

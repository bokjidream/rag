---
glob: "src/pipeline/**"
---

# 파이프라인 규칙 — src/pipeline/ 작업 시 적용

## 임베딩

- 모델: `EMBEDDING_MODEL` 환경변수 사용 (기본값: `jhgan/ko-sroberta-multitask`)
- HuggingFace 모델은 로컬 캐시 활용, 매번 다운로드 금지

## 에러 처리

- 청크 임베딩 실패 시: 해당 청크만 skip, 나머지 계속 진행
- 실패한 청크는 `logger.warning()`으로 기록

## 출력

- 청킹 결과 메타데이터는 `data/processed/` 에 저장
- ChromaDB persist 경로: `CHROMA_PERSIST_DIR` 환경변수

## ChromaDB

- 컬렉션 이름은 상수로 관리
- 동일 문서 재인덱싱 시 중복 처리 전략 명시 (upsert 사용)

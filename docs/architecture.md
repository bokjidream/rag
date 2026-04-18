# 아키텍처

## 전체 시스템 맥락

```
사용자
  ↓
웹 (Next.js + Supabase)
  ↓
LangGraph 오케스트레이터
  ↓
멀티 에이전트 (인터뷰 / 수급분석 / 서류안내 / 초안작성 / 리포트)
  ↓
RAG (이 레포) ← 수급 분석 에이전트가 호출
  ↓
공공데이터 API → ChromaDB
```

## 이 레포의 데이터 흐름

```
공공데이터 API (apis.data.go.kr)
  → src/crawler/      # HTTP requests로 복지 서비스 수집 (391개)
  → data/raw/         # 원본 JSON 저장
  → src/pipeline/     # 청킹·임베딩·ChromaDB 인덱싱
  → data/processed/   # 메타데이터 저장
  → src/retriever/    # FastAPI POST /search, GET /detail/{servId}
  → LangGraph 팀
```

> **주의**: 크롤러는 Playwright(브라우저)가 아닌 `requests` 라이브러리로 공공데이터 API를 호출한다.

## 레이어별 책임

| 레이어 | 모듈 | 책임 | 입력 | 출력 |
|--------|------|------|------|------|
| 크롤러 | `src/crawler/` | 공공데이터 API 호출 및 수집 | - | `data/raw/*.json` |
| 파이프라인 | `src/pipeline/` | 청킹 + 임베딩 + ChromaDB 인덱싱 | `data/raw/` | ChromaDB 컬렉션 |
| 리트리버 | `src/retriever/` | 벡터 검색 + FastAPI 엔드포인트 | 사용자 정보 JSON | JSON 응답 |

## 공공데이터 API

| 종류 | URL |
|------|-----|
| 목록 | `GET https://apis.data.go.kr/B554287/NationalWelfareInformationsV001/NationalWelfarelistV001` |
| 상세 | `GET https://apis.data.go.kr/B554287/NationalWelfareInformationsV001/NationalWelfaredetailedV001` |

- 인증: `WELFARE_API_KEY` 환경변수
- 총 서비스 수: **391개**

## ChromaDB 스키마

```python
# 검색용 벡터 텍스트
document = f"""
서비스명: {servNm}
대상자: {trgterIndvdlArray}
지원대상상세: {tgtrDtlCn}
선정기준: {slctCritCn}
지원내용: {alwServCn}
개요: {wlfareInfoOutlCn}
"""

# 메타데이터 (검색 결과와 함께 반환)
metadata = {
    "servId": servId,
    "servNm": servNm,
    "trgterIndvdlArray": trgterIndvdlArray,
    "alwServCn": alwServCn,
    "onapPsbltYn": onapPsbltYn,
    "rprsCtadr": rprsCtadr,
    "apply_method": apply_method,
    "servDtlLink": servDtlLink,
}

# ID: servId 그대로 사용 (예: "WLF00000035")
```

## 쿼리 변환 로직

사용자 정보 JSON → 벡터 검색 쿼리 문자열 변환:

```python
def json_to_query(user_info: dict) -> str:
    parts = []
    if user_info.get("age"):
        parts.append(f"{user_info['age']}세")
    if user_info.get("household_size"):
        parts.append(f"{user_info['household_size']}인가구")
    if user_info.get("income_level") == 0:
        parts.append("소득없음")
    elif user_info.get("income_level"):
        parts.append("저소득")
    if user_info.get("disability"):
        parts.append("장애인")
    return " ".join(parts)
    # 예시: "65세 1인가구 소득없음"
```

## 환경변수

| 변수 | 용도 |
|------|------|
| `WELFARE_API_KEY` | 공공데이터 포털 인증키 |
| `EMBEDDING_MODEL` | HuggingFace 임베딩 모델 (jhgan/ko-sroberta-multitask) |
| `CHROMA_PERSIST_DIR` | ChromaDB 저장 경로 |
| `RAG_API_PORT` | FastAPI 서버 포트 (기본 8002) |

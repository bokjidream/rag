# PRD: BokjiDream RAG

## 목표

복지 지침서 PDF와 공공데이터 API를 수집·인덱싱하여, LangGraph ② 수급분석 에이전트가 사용자 조건(나이·소득·가구)에 맞는 복지 서비스를 정확하게 검색할 수 있는 RAG 파이프라인을 제공한다.

## 사용자 (내부 인터페이스)

직접 최종 사용자를 상대하지 않는다. **LangGraph 오케스트레이터**가 이 RAG API를 호출하는 유일한 클라이언트다.

## 핵심 기능

1. **데이터 수집**: 복지로 크롤링(Playwright), 공공데이터포털 API, 복지 지침서 PDF 파싱
2. **벡터 인덱싱**: 수집 문서를 청크 분할 → 한국어 임베딩 → ChromaDB 저장
3. **검색 API**: 사용자 조건 쿼리를 받아 관련 복지 서비스 목록을 JSON으로 반환 (FastAPI)
4. **배치 갱신**: AWS Lambda 기반 주기적 크롤링 및 ChromaDB 업데이트

## API 인터페이스 (LangGraph 팀과 합의된 JSON 교환 형식)

원천 데이터 수집 흐름: 공공데이터포털 복지서비스 API (XML) → 파싱 → ChromaDB 인덱싱. 이 흐름은 내부 크롤러/파이프라인에서 처리하며 LangGraph에는 노출되지 않는다.

### API 1: 서비스 상세 조회 (LangGraph 전용)

```
GET /welfare/{serv_id}

응답: ChromaDB에서 서빙 (공공데이터포털 직접 호출 아님)
{
  "serv_id": "WLF00000035",
  "serv_nm": "서비스명",
  "serv_dgst": "서비스 개요",
  "tgtr_dtl_cn": "수급 대상 상세...",
  "slct_crit_cn": "선정 기준...",
  "alw_serv_cn": "서비스 내용...",
  "sprt_cyc_nm": "년",
  "srv_pvsn_nm": "현금지급",
  "trgter_indvdl": ["저소득"],
  "intrs_thema": ["주거", "생활지원"],
  "application_url": "https://bokjiro.go.kr/...",
  "required_documents": [],
  "application_fields": []
}
```

- `application_url`: 공공데이터 API `servDtlLink` 필드 매핑
- `required_documents`, `application_fields`: MVP에서 빈 배열 반환. 복지로 크롤링 추가 시 채워질 예정

### API 2: RAG 검색 — 요약 목록 반환 (LangGraph 전용)

```
POST /welfare/search

요청:
{
  "age": 65,
  "income_level": "저소득",        # "기초생활수급자" | "차상위계층" | "저소득" | "일반"
  "household_size": 1,             # 가구원 수
  "marital_status": "미혼",        # "미혼" | "기혼" | "이혼" | "사별"
  "has_children": false,           # 미성년 자녀 유무
  "disability": false,
  "disability_severity": null,     # "경증" | "중증" | null (disability=true일 때만)
  "employment_status": "실업",     # "취업" | "실업" | "비경제활동"
  "region": "서울",                # optional. 쿼리 텍스트에 포함되어 벡터 검색에 활용
  "top_k": 5
}

응답: 요약 필드만 반환 (상세 텍스트 제외)
{
  "results": [
    {
      "serv_id": "WLF00000035",
      "serv_nm": "서비스명",
      "serv_dgst": "서비스 개요",
      "department": "국토교통부",
      "score": 0.87,
      "trgter_indvdl": ["저소득"],
      "intrs_thema": ["주거", "생활지원"]
    }
  ]
}
```

- `age`, `income_level` 외 나머지 필드는 optional
- `income_level` enum: LangGraph `UserProfile`과 동일한 4구간
- `disability_severity`: `disability=true`일 때만 의미 있음
- `region`: 메타데이터 필터 불가, 쿼리 텍스트에 포함하여 벡터 검색으로 soft 매칭. MVP 이후 hard filter 전환 가능
- `department`: 공공데이터 API `jurMnofNm` 필드 매핑
- `eligibility_reason`: RAG 응답에 포함하지 않음 — LangGraph LLM이 유저 프로파일 + 검색 결과 기반으로 생성
- 상세 텍스트는 응답에 포함하지 않음 — 유저가 서비스 선택 시 API 1로 별도 조회
- MVP 기준 스펙. 항목 추가/삭제 시 LangGraph 팀(재표형)과 사전 협의

### LangGraph 호출 흐름

```
1단계: POST /welfare/search  → 유저 조건으로 관련 서비스 top-k 요약 목록 수신 (API 2)
2단계: GET /welfare/{serv_id} → 유저가 선택한 서비스의 상세 내용 수신 (API 1)
```

serv_id는 항상 POST /welfare/search 응답에서 획득. 직접 호출 불가.

> ⚠ 필드명·응답 구조 변경 시 LangGraph 팀(재표형)과 반드시 사전 협의

## MVP 제외 사항

- 실시간 크롤링 (배치로 충분)
- 사용자 인증 / 로그인
- 검색 이력 저장 (Supabase는 웹 팀 담당)
- Llama 쉬운말 변환 (③ 서류안내 에이전트 담당)

## 성공 기준

- 수급 조건 쿼리에 대해 관련 서비스 top-5 정확도 80% 이상
- 검색 응답 시간 2초 이내
- ChromaDB 인덱스에 복지 서비스 500건 이상 적재

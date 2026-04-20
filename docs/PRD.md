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

원천 데이터는 공공데이터포털 복지서비스 API (XML) → 파싱 후 ChromaDB 인덱싱.

### API 1: 목록 조회

```
GET /welfare/list
Query params: {검색 조건} ← 이번 프로젝트에서 내가 설계하는 핵심 부분

응답:
{
  "total_count": 391,
  "results": [
    {
      "serv_id": "WLF00000023",
      "serv_nm": "농어가목돈마련저축 저축장려금 지급",
      "serv_dgst": "...",
      "jur_mnof_nm": "금융위원회",
      "intrs_thema": ["서민금융"],
      "onap_psblt_yn": "N",
      "sprt_cyc_nm": "1회성",
      "srv_pvsn_nm": "현금지급",
      "serv_dtl_link": "https://..."
    }
  ]
}
```

### API 2: 서비스 상세 조회 (LangGraph 전용)

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
  "intrs_thema": ["주거", "생활지원"]
}
```

### API 3: RAG 검색 — 요약 목록 반환 (LangGraph 전용)

```
POST /welfare/search

요청:
{
  "age": 65,
  "income_level": "저소득",   # "기초수급" | "차상위" | "저소득" | "일반"
  "disability": false,
  "family_type": null,         # "한부모" | "다자녀" | null
  "pregnant": false,
  "top_k": 5
}

응답: 요약 필드만 반환 (상세 텍스트 제외)
{
  "results": [
    {
      "serv_id": "WLF00000035",
      "serv_nm": "서비스명",
      "serv_dgst": "서비스 개요",
      "score": 0.87,
      "trgter_indvdl": ["저소득"],
      "intrs_thema": ["주거", "생활지원"]
    }
  ]
}
```

- LangGraph가 유저 조건 구조체를 전달하면, RAG 레이어가 벡터 쿼리 + 메타데이터 필터로 변환하여 검색
- `age`, `income_level` 외 나머지 필드는 optional
- 상세 텍스트(`tgtrDtlCn`, `slctCritCn`, `alwServCn`)는 응답에 포함하지 않음 — 유저가 특정 서비스 선택 시 API 2로 별도 조회
- MVP 기준 스펙. 항목 추가/삭제 시 LangGraph 팀(재표형)과 사전 협의

### LangGraph 호출 흐름

```
1단계: POST /welfare/search → 유저 조건으로 관련 서비스 top-k 요약 목록 수신
2단계: GET /welfare/{serv_id} → 유저가 선택한 서비스의 상세 내용 수신
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

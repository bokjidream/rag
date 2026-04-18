# API 계약 — LangGraph 팀 인터페이스

## 엔드포인트

| 메서드 | 경로 | 용도 |
|--------|------|------|
| `POST` | `/search` | 사용자 정보로 관련 복지 서비스 목록 반환 |
| `GET` | `/detail/{servId}` | servId로 서비스 상세 정보 반환 |

---

## POST /search

### 요청 (LangGraph → RAG)

```json
{
  "age": 65,
  "household_size": 1,
  "income_level": 0,
  "disability": false,
  "region": "서울"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `age` | int | 나이 |
| `household_size` | int | 가구원 수 |
| `income_level` | int | 소득 수준 (0=없음) |
| `disability` | bool | 장애 여부 |
| `region` | str | 지역 |

### 응답 (RAG → LangGraph)

```json
{
  "results": [
    {
      "servId": "WLF00000035",
      "servNm": "기초연금",
      "trgterIndvdlArray": "저소득,노인",
      "alwServCn": "월 최대 334,000원",
      "onapPsbltYn": "Y",
      "rprsCtadr": "129",
      "apply_method": "읍/면/동 주민센터",
      "servDtlLink": "https://bokjiro.go.kr/...",
      "score": 0.95
    }
  ]
}
```

---

## GET /detail/{servId}

### 응답 (RAG → LangGraph)

```json
{
  "servId": "WLF00000035",
  "servNm": "기초연금",
  "tgtrDtlCn": "만 65세 이상...",
  "slctCritCn": "소득 하위 70%...",
  "alwServCn": "월 최대 334,000원 지급",
  "apply_method": "읍/면/동 주민센터 방문 신청",
  "onapPsbltYn": "Y",
  "rprsCtadr": "129",
  "servDtlLink": "https://bokjiro.go.kr/..."
}
```

---

## 필드 규칙

- 기존 필드 **삭제·변경 금지** (LangGraph 팀과 사전 협의 필요)
- 필드 추가는 협의 없이 가능

## 에러 응답 (변경 금지)

```json
{
  "error": "string",
  "detail": "string"
}
```

모든 HTTP 에러(4xx, 5xx)에서 이 형식 유지.

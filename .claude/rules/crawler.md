---
glob: "src/crawler/**"
---

# 크롤러 규칙 — src/crawler/ 작업 시 적용

## 수집 방식

- Playwright(브라우저) 사용 금지 — 공공데이터 REST API를 `requests`로 호출
- 비동기: `httpx` 또는 `asyncio` 기반 async HTTP 클라이언트 사용

## 공공데이터 API

```
목록: GET https://apis.data.go.kr/B554287/NationalWelfareInformationsV001/NationalWelfarelistV001
상세: GET https://apis.data.go.kr/B554287/NationalWelfareInformationsV001/NationalWelfaredetailedV001
```

- 인증키: `WELFARE_API_KEY` 환경변수 (하드코딩 금지)
- 총 서비스 수: 391개

## 수집 전략

1. 목록 API로 전체 `servId` 수집
2. 각 `servId`로 상세 API 호출
3. 실패한 `servId`는 `logger.warning()` 후 `continue` (전체 중단 금지)

## 출력

- `data/raw/welfare_services.json` 에 전체 저장
- 각 서비스는 `servId`를 키로 구분

# Step 4: crawler

## 읽어야 할 파일

먼저 아래 파일들을 읽고 설계 의도를 파악하라:

- `docs/ARCHITECTURE.md`
- `docs/ADR.md` (ADR-004, ADR-005)
- `src/models/welfare.py` (WelfareRaw 모델 확인)
- `phases/0-mvp/index.json` (step 3 summary 확인)

## 작업

공공데이터포털 복지서비스 API를 호출하여 `WelfareRaw` 객체 리스트를 반환하는 크롤러를 구현한다.
**외부 API 호출은 이 레이어에서만 수행한다.**

### 구현할 파일

#### 1. `src/crawler/client.py` — HTTP 클라이언트

```python
from __future__ import annotations
import httpx

def build_client(timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)
```

- `httpx.AsyncClient`를 생성하는 팩토리 함수
- 직접 인스턴스를 모듈 레벨에 두지 않는다 (테스트 격리를 위해)

#### 2. `src/crawler/welfare_list.py` — 목록 조회

공공데이터포털 복지서비스 목록 API (`servList`)를 호출한다.
XML 파싱은 **표준 라이브러리 `xml.etree.ElementTree`만 사용**한다 (`lxml` 설치 불필요).

```python
import xml.etree.ElementTree as ET

async def fetch_welfare_list(
    api_key: str,
    page: int = 1,
    per_page: int = 100,
) -> list[dict[str, Any]]:
    """공공데이터포털 목록 API 호출 → 파싱된 dict 리스트 반환.
    
    api_key는 호출부(collect_all)에서 환경변수를 읽어 전달한다.
    이 함수 내부에서 환경변수를 직접 읽지 않는다.
    """
```

- XML 응답 루트 구조: `<servList> → <servInfo> → <servId>, <servNm>, ...` 형식
- `trgterIndvdlArray`, `intrsThemaArray`는 콤마 구분 문자열 → `list[str]`로 파싱
  - 빈 문자열이나 None이면 빈 리스트로 처리

#### 3. `src/crawler/welfare_detail.py` — 상세 조회

공공데이터포털 복지서비스 상세 API (`servDtl`)를 호출한다.
XML 파싱도 `xml.etree.ElementTree`를 사용한다.

```python
import xml.etree.ElementTree as ET

async def fetch_welfare_detail(
    serv_id: str,
    api_key: str,
) -> dict[str, Any]:
    """공공데이터포털 상세 API 호출 → 파싱된 dict 반환.
    
    api_key는 호출부(collect_all)에서 전달한다.
    """
```

#### 4. `src/crawler/collect.py` — 배치 수집 진입점

환경변수 `PUBLIC_DATA_API_KEY`는 **이 함수에서 읽는다**.
`fetch_welfare_list`, `fetch_welfare_detail`에는 파라미터로 전달한다.

```python
import os

async def collect_all(max_pages: int = 10) -> list[WelfareRaw]:
    """목록 API 전체 페이지 수집 + 상세 API로 보강 → WelfareRaw 리스트 반환.
    
    환경변수 PUBLIC_DATA_API_KEY가 없으면 ValueError 발생.
    """
    api_key = os.environ.get("PUBLIC_DATA_API_KEY")
    if not api_key:
        raise ValueError("PUBLIC_DATA_API_KEY 환경변수가 설정되지 않았습니다.")
    ...
```

- 목록 API를 페이지네이션으로 전수 수집
  - 종료 조건: 응답 항목 수 < `per_page` 이면 마지막 페이지로 간주하고 루프 종료
  - `max_pages` 초과 시에도 종료 (안전장치)
- 각 항목마다 상세 API 호출하여 `tgtr_dtl_cn`, `slct_crit_cn`, `alw_serv_cn` 보강
- 최종적으로 `WelfareRaw` 객체 리스트를 반환 (pipeline에 전달)
- API 오류 시 해당 항목만 스킵하고 계속 진행. 아래 예외를 명시적으로 처리:
  ```python
  except (httpx.HTTPError, httpx.TimeoutException,
          ET.ParseError, KeyError, ValueError) as e:
      # 로그 출력 시 api_key가 URL에 포함되지 않도록 주의
      # httpx 로깅 레벨을 WARNING 이상으로 유지하거나 log_config 미설정
      logger.warning("항목 스킵: %s", str(e))
      continue
  ```

### 공공데이터포털 API 엔드포인트

```
목록: GET https://apis.data.go.kr/B554287/NationalWelfareInformationService/NationalWelfarelistInquiry
상세: GET https://apis.data.go.kr/B554287/NationalWelfareInformationService/NationalWelfareDetailInquiry
```

공통 파라미터: `serviceKey={api_key}`, `callTp=L` (목록) / `callTp=D` (상세), `srvcId={serv_id}`

### XML 필드 매핑 (목록 → WelfareRaw)

| XML 필드 | WelfareRaw 필드 |
|----------|----------------|
| `servId` | `serv_id` |
| `servNm` | `serv_nm` |
| `servDgst` | `serv_dgst` |
| `jurMnofNm` | `jur_mnof_nm` |
| `trgterIndvdlArray` (콤마 구분) | `trgter_indvdl` (list) |
| `intrsThemaArray` (콤마 구분) | `intrs_thema` (list) |
| `sprtCycNm` | `sprt_cyc_nm` |
| `srvPvsnNm` | `srv_pvsn_nm` |
| `servDtlLink` | `serv_dtl_link` |

### XML 필드 매핑 (상세 → WelfareRaw 보강)

| XML 필드 | WelfareRaw 필드 |
|----------|----------------|
| `tgtrDtlCn` | `tgtr_dtl_cn` |
| `slctCritCn` | `slct_crit_cn` |
| `alwServCn` | `alw_serv_cn` |

### 주의사항

- `src/models/welfare.py`의 `WelfareRaw`만 임포트. 다른 레이어 임포트 금지
- 개인정보(나이, 소득 등 사용자 조건)는 이 레이어에서 처리하지 않는다
- `PUBLIC_DATA_API_KEY`가 없으면 `ValueError` 발생 (앱 시작 시 early validation)
- **보안**: `api_key`를 로그에 출력하지 마라. httpx는 기본적으로 요청 URL을 로깅할 수 있으며,
  URL에 `serviceKey={api_key}` 가 포함되므로 `logging.getLogger("httpx").setLevel(logging.WARNING)`
  설정이 필요하다. `logger.warning(...)` 호출 시 URL 전체나 `api_key` 변수를 포함하지 마라.
- httpx.AsyncClient는 사용 후 반드시 닫는다:
  ```python
  async with build_client() as client:
      response = await client.get(url, params=params)
  ```

## Acceptance Criteria

```bash
mypy src/crawler/                                                      # 타입 오류 없음
pytest tests/integration/test_crawler.py -v -m "not integration"     # mock 테스트만 실행 (API 키 불필요)
ruff check src/crawler/                                                # 린트 오류 없음
```

테스트 파일 `tests/integration/test_crawler.py`도 함께 작성한다.
최소한 아래 케이스를 커버해야 한다:

- `fetch_welfare_list()` XML 응답 mock → `WelfareRaw` 필드 정상 파싱
- `trgterIndvdlArray` 콤마 구분 문자열 → `list[str]` 변환 검증
- `PUBLIC_DATA_API_KEY` 없을 때 `ValueError` 발생
- `collect_all()` API 오류 시 해당 항목 스킵하고 계속 진행

> 실제 외부 API를 호출하는 테스트는 `@pytest.mark.integration` 마크를 붙이고,
> CI에서는 기본 제외한다. Mock 기반 테스트만 기본 실행.

## 검증 절차

1. 위 AC 커맨드를 순서대로 실행한다.
2. 아키텍처 체크리스트:
   - 외부 API 호출이 `src/crawler/`에만 있는가?
   - `WelfareRaw` 모든 필드가 올바르게 매핑되는가?
   - `db/`, `embedding/`, `retriever/`, `api/`를 임포트하지 않는가?
3. 결과에 따라 `phases/0-mvp/index.json`의 step 4를 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "src/crawler/ 구현 완료 — fetch_welfare_list/detail, collect_all 함수, WelfareRaw XML 파싱 완료"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 (`PUBLIC_DATA_API_KEY` 발급 필요 등) → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- Playwright를 이 단계에서 도입하지 마라 (ADR-005: MVP는 공공 API만)
- 사용자 개인정보(나이, 소득 등)를 이 레이어에서 처리하지 마라
- 크롤러가 ChromaDB에 직접 접근하지 마라 — 반드시 `WelfareRaw` 리스트를 반환하고 pipeline에 위임

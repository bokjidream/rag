# Step 1: models

## 읽어야 할 파일

먼저 아래 파일들을 읽고 설계 의도를 파악하라:

- `docs/ARCHITECTURE.md`
- `docs/ADR.md`
- `docs/PRD.md`
- `phases/0-mvp/index.json` (step 0 summary 확인)

## 작업

`src/models/welfare.py`를 생성한다. 이 파일은 프로젝트 전체에서 임포트되는
**유일한 Pydantic 스키마 모음**이다. 이후 모든 step이 이 파일에 의존하므로
필드명 하나하나를 PRD와 정확히 일치시켜야 한다.

### 구현할 모델

#### 1. SearchRequest — POST /welfare/search 요청 바디

```python
class SearchRequest(BaseModel):
    age: int
    income_level: Literal["기초생활수급자", "차상위계층", "저소득", "일반"]
    household_size: int | None = None          # 가구원 수
    marital_status: Literal["미혼", "기혼", "이혼", "사별"] | None = None
    has_children: bool | None = None           # 미성년 자녀 유무
    disability: bool = False
    disability_severity: Literal["경증", "중증"] | None = None  # disability=True일 때만
    employment_status: Literal["취업", "실업", "비경제활동"] | None = None
    region: str | None = None   # 쿼리 텍스트에 포함. 메타데이터 필터 아님
    top_k: int = 5
```

- `age`, `income_level`만 필수. 나머지는 모두 optional
- `disability_severity`는 `disability=True`일 때만 의미 있음 (검증은 Step 7 api-layer에서)
- `income_level` enum: LangGraph `UserProfile`과 동일한 4구간

#### 2. SearchResult — POST /welfare/search 응답 항목 (요약)

```python
class SearchResult(BaseModel):
    serv_id: str
    serv_nm: str
    serv_dgst: str
    department: str            # 공공데이터 API jurMnofNm 필드 매핑
    score: float
    trgter_indvdl: list[str]
    intrs_thema: list[str]
```

#### 3. SearchResponse — POST /welfare/search 응답 전체

```python
class SearchResponse(BaseModel):
    results: list[SearchResult]
```

#### 4. WelfareDetail — GET /welfare/{serv_id} 응답

```python
class WelfareDetail(BaseModel):
    serv_id: str
    serv_nm: str
    serv_dgst: str
    tgtr_dtl_cn: str
    slct_crit_cn: str
    alw_serv_cn: str
    sprt_cyc_nm: str
    srv_pvsn_nm: str
    trgter_indvdl: list[str]
    intrs_thema: list[str]
    application_url: str       # 공공데이터 API servDtlLink 필드 매핑
    required_documents: list[str] = []   # MVP: 빈 배열. 복지로 크롤링 추가 시 채워질 예정
    application_fields: list[str] = []   # MVP: 빈 배열
```

#### 5. WelfareRaw — 공공데이터 API 파싱 결과 (crawler → pipeline 전달용)

crawler가 XML을 파싱한 뒤 pipeline에 넘길 때 사용하는 내부 모델이다.
LangGraph에 노출되지 않는다.

```python
class WelfareRaw(BaseModel):
    serv_id: str
    serv_nm: str
    serv_dgst: str
    jur_mnof_nm: str
    trgter_indvdl: list[str]   # XML의 trgterIndvdlArray를 파싱
    intrs_thema: list[str]     # XML의 intrsThemaArray를 파싱
    sprt_cyc_nm: str
    srv_pvsn_nm: str
    serv_dtl_link: str
    # 상세 API 필드 (목록 API에는 없음, 기본값 빈 문자열)
    tgtr_dtl_cn: str = ""
    slct_crit_cn: str = ""
    alw_serv_cn: str = ""
```

### 주의사항

- `trgterIndvdlArray`, `intrsThemaArray`는 XML에서 콤마 구분 문자열로 온다
  (`"저소득,노인"`). `WelfareRaw`에서는 이미 파싱된 `list[str]`로 받는다.
  파싱 로직은 Step 4(crawler)에서 구현한다.
- 모든 필드에 타입 힌트를 명시한다 (mypy strict 통과 필수).
- `from __future__ import annotations`를 파일 상단에 추가한다 (Python 3.9 호환).

## Acceptance Criteria

```bash
mypy src/models/                          # 타입 오류 없음
pytest tests/unit/test_models.py -v      # 모든 테스트 통과
ruff check src/models/                   # 린트 오류 없음
```

테스트 파일 `tests/unit/test_models.py`도 함께 작성한다.
최소한 아래 케이스를 커버해야 한다:

- `SearchRequest` 필수 필드(`age`, `income_level`) 누락 시 ValidationError 발생
- `income_level` 허용값 외 값 입력 시 ValidationError 발생
- `marital_status` 허용값 외 값 입력 시 ValidationError 발생
- `employment_status` 허용값 외 값 입력 시 ValidationError 발생
- `SearchResult`에 `department` 포함하여 정상 생성
- `WelfareDetail`에 `application_url`, `required_documents`, `application_fields` 포함하여 정상 생성
- `WelfareRaw` 정상 생성

## 검증 절차

1. 위 AC 커맨드를 순서대로 실행한다.
2. 아키텍처 체크리스트:
   - `src/models/welfare.py` 하나의 파일에 모든 모델이 있는가?
   - PRD의 필드명과 100% 일치하는가? (`serv_id`, `serv_nm` 등 snake_case)
   - LangGraph 노출 모델(`SearchRequest`, `SearchResult`, `SearchResponse`, `WelfareDetail`)과
     내부 모델(`WelfareRaw`)이 명확히 구분되는가?
3. 결과에 따라 `phases/0-mvp/index.json`의 step 1을 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "src/models/welfare.py 생성 — SearchRequest/SearchResult/SearchResponse/WelfareDetail/WelfareRaw 정의 완료"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- 비즈니스 로직을 구현하지 마라. Pydantic 모델 정의만 한다.
- 다른 `src/` 모듈을 임포트하지 마라. `models/`는 의존성이 없어야 한다.
- PRD에 없는 필드를 임의로 추가하지 마라. 변경이 필요하면 PRD를 먼저 수정한다.

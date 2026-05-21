# RAG Search Quality Plan

## Current Goal

검색 품질 개선은 청킹 변경부터 시작하지 않는다. 먼저 실사용 시나리오 평가셋을
고정하고, 현재 구현의 baseline을 측정한 뒤 같은 평가셋으로 청킹/데이터 개선 효과를
비교한다.

## Evaluation Baseline

Command:

```bash
.venv/bin/python scripts/evaluate_search.py --verbose
```

Baseline measured on the current persistent Chroma index after expanding the
real-index evaluation set to 100 cases:

- Index: 570 chunks / 413 services
- Cases: 100
- Validation errors: 0
- Recall@1: 0.390
- Recall@3: 0.560
- Recall@5: 0.680
- Recall@10: 0.780
- MRR: 0.503
- Exclusion pass rate: 0.950
- Avg latency: 48.9 ms/case
- P95 latency: 58.6 ms/case
- Returned eligibility status distribution:
  - `likely`: 782 / 1000 results (0.782)
  - `needs_more_info`: 218 / 1000 results (0.218)
  - `unlikely`: 0 / 1000 results (0.000)
- Exit code: 1, because this is a baseline quality failure, not an
  `EVAL_CASE_VALIDATION_ERROR`

`unlikely` is zero in the returned distribution because `search_welfare()`
filters those candidates before building the response. Counting filtered
`unlikely` candidates requires a separate debug/instrumentation path.

Latency is measured around each `search_welfare()` call after constructing the
embedder, so it excludes model startup time and includes query embedding,
Chroma lookup, ranking, and eligibility filtering.

## Known Failure Cases

1. `영유아_자녀_일반`
   - Expected child-care services appear at rank 8 or lower.
   - Current top results over-match 한부모/자립준비청년/청소년부모 terms.
   - Likely issue: child-care intent is not separated enough from broader child/youth welfare terms.

2. `차상위_중년`
   - Expected services are not in top 10.
   - Current top results include some reasonable 차상위/low-income services, but not the expected set.
   - Likely issue: expected set may need review, and 차상위-specific text may need better chunking/metadata.

3. `61세_저소득_비장애_자녀없음`
   - Expected services are found, but `노인일자리 및 사회활동 지원사업` appears in top 10.
   - Likely issue: under-65 senior-adjacent false positives need stricter eligibility filtering or better section scoring.

## 100-Case Baseline Failure Types

The 100-case baseline has 33 failing cases. Five cases fail because an
`excluded_id` appears in the returned top 10:

- `61세_저소득_비장애_자녀없음`: `WLF00001155` 노인일자리 및 사회활동 지원사업
- `61세_일반_노인전용오탐방지`: `WLF00001155` 노인일자리 및 사회활동 지원사업
- `64세_일반_노후전환오탐방지`: `WLF00001155` 노인일자리 및 사회활동 지원사업
- `미혼_청년_한부모오탐방지`: `WLF00001068` 한부모가족 아동양육비 지원
- `농어업정보없음_일반_오탐방지`: `WLF00001099` 농업인건강보험료지원

Representative expected-service misses:

- Child and childcare queries often over-match 한부모/청소년부모/자립준비청년
  services: `영유아_자녀_일반`, `5세_일반_영유아`,
  `5세_저소득_영유아건강`, `아동_저소득_교육문화`.
- Theme-specific low-income services are not separated well by current chunks:
  `저소득_에너지`, `저소득_문화`, `저소득_의료비`,
  `기초수급_의료`.
- General profile queries are underspecified and drift toward broad
  care/housing results: `일반_중장년_특수대상오탐방지`,
  `북한이탈정보없음_일반_오탐방지`,
  `다문화정보없음_일반_오탐방지`,
  `청소년부모정보없음_성인_오탐방지`,
  `중장년_일반_건강생활`.
- Age-boundary behavior is weak around 61/64/65:
  under-65 cases surface senior employment/care, while `65세_일반_노후준비`
  misses general 노후준비/장기요양 candidates.
- Pregnancy and youth finance cases are dominated by housing/childcare
  matches: `임산부_일반`, `청소년산모_저소득`,
  `19세_일반_청년금융`, `청년_취업_금융`,
  `청년_일반_교육대출`.
- Disability cases need better section weighting for non-income, non-housing
  intents: `경증장애인_일반_생활`, `장애인_중장년_창업`.

## Next Development Sequence

1. Keep this evaluation script as the baseline gate.
2. Implement section-aware chunking:
   - `summary`: service name + summary
   - `target`: 지원대상
   - `criteria`: 선정기준
   - `benefit`: 지원내용
   - `application`: 신청방법
   - `documents`: 필요서류/신청서식
3. Store `chunk_section` in Chroma metadata and aggregate final results by `serv_id`.
4. Re-run the same evaluation script and compare:
   - Recall@5
   - MRR
   - Exclusion pass rate
   - Avg/P95 latency
   - Returned `likely` / `needs_more_info` distribution
   - The three known failure cases above
5. Only after the chunking result is measurable, expand/clean the data source.
6. Dockerize the RAG API after search behavior stabilizes enough for demo/deploy.

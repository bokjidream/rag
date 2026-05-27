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

## Section-Aware Chunking Experiment

Build command:

```bash
.venv/bin/python scripts/build_section_aware_index.py
```

Evaluation command:

```bash
.venv/bin/python scripts/evaluate_search.py --chroma-persist-dir data/chroma-section-aware --collection welfare_services_section_aware
```

The default baseline evaluation copies `data/chroma` to a temporary directory
before opening Chroma, because the persistent client can update SQLite state even
for read-only quality checks.

Index output:

- Source index: 570 chunks / 413 services
- Section-aware index: 2,424 chunks / 413 services
- Target path: `data/chroma-section-aware`
- Target collection: `welfare_services_section_aware`
- Section counts:
  - `summary`: 413
  - `target`: 426
  - `criteria`: 438
  - `benefit`: 429
  - `application`: 414
  - `documents`: 304

Section-aware chunks are generated from the same baseline Chroma metadata,
deduped by `serv_id`. The baseline `data/chroma` index remains the default
evaluation target and is not overwritten. Section chunks include service name
and a section label in the embedded text while preserving the original metadata
fields for `SearchResult` / `WelfareDetail` compatibility.

Comparison uses the same 100 cases. The baseline accuracy metrics are unchanged;
latency was re-measured in the same session because it varies by runtime state.

Comparison:

| Metric | Baseline | Section-aware |
| --- | ---: | ---: |
| Recall@1 | 0.390 | 0.380 |
| Recall@3 | 0.560 | 0.550 |
| Recall@5 | 0.680 | 0.680 |
| Recall@10 | 0.780 | 0.780 |
| MRR | 0.503 | 0.506 |
| Exclusion pass rate | 0.950 | 0.970 |
| Avg latency | 45.7 ms | 54.3 ms |
| P95 latency | 57.2 ms | 59.4 ms |
| `likely` distribution | 782 / 1000 (0.782) | 764 / 1000 (0.764) |
| `needs_more_info` distribution | 218 / 1000 (0.218) | 236 / 1000 (0.236) |
| `unlikely` distribution | 0 / 1000 (0.000) | 0 / 1000 (0.000) |

Result:

- Section-aware chunking is not a stable top-5 improvement by itself.
- The main wins are fewer excluded-service leaks and a slightly better MRR.
- The tradeoff is higher latency from adaptive candidate fetch over a larger
  chunk index.
- Remaining failures suggest the next improvement should focus on query intent
  and section weighting, not more raw data volume yet.

## Goal 4 Reranking / Filtering Pass

Command:

```bash
.venv/bin/python scripts/evaluate_search.py --chroma-persist-dir data/chroma-section-aware --collection welfare_services_section_aware
```

Implementation summary:

- `SearchRequest` is converted to a lightweight `QueryIntent` for reranking input.
- Section-aware evaluation fetches a wider raw candidate set, groups chunks by
  `serv_id`, then ranks a single service result using section best scores.
- `target` and `criteria` evidence receive the highest section weight; `benefit`
  is medium weight; `application` and `documents` are intentionally weak so
  procedural chunks do not dominate final service rank.
- Negative condition penalties are soft score penalties. Hard exclusion remains
  limited to `evaluate_eligibility(...).status == "unlikely"`.
- The baseline collection path keeps the original best-chunk ranking behavior.

Latest comparison:

| Metric | Baseline latest | Section-aware before | Section-aware after |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.390 | 0.380 | 0.350 |
| Recall@3 | 0.560 | 0.550 | 0.580 |
| Recall@5 | 0.680 | 0.680 | 0.680 |
| Recall@10 | 0.780 | 0.780 | 0.790 |
| MRR | 0.503 | 0.506 | 0.482 |
| Exclusion pass rate | 0.950 | 0.970 | 1.000 |
| Avg latency | 45.1 ms | 56.6 ms | 59.8 ms |
| P95 latency | 48.9 ms | 63.7 ms | 67.9 ms |
| `likely` distribution | 782 / 1000 (0.782) | 764 / 1000 (0.764) | 773 / 1000 (0.773) |
| `needs_more_info` distribution | 218 / 1000 (0.218) | 236 / 1000 (0.236) | 227 / 1000 (0.227) |
| `unlikely` distribution | 0 / 1000 (0.000) | 0 / 1000 (0.000) | 0 / 1000 (0.000) |

Representative improvements:

- `61세_일반_노인전용오탐방지`, `64세_일반_노후전환오탐방지`,
  `미혼_청년_한부모오탐방지`, and `농어업정보없음_일반_오탐방지`
  no longer return excluded services in the section-aware top 10.
- `영유아_자녀_일반` improved from rank 6 to rank 5.
- `6세_저소득_아동돌봄` improved from rank 8 to rank 4.
- `65세_일반_노후준비` improved from rank 8 to rank 5.

Representative regressions:

- `저소득_임산부` moved from rank 4 to rank 6 because broad housing/family
  services still receive very high profile boosts.
- `저소득_금융` moved from rank 7 to rank 8; the current request has no topic
  field to distinguish finance from general low-income housing or employment.
- `한부모_저소득_교육` improved from rank 9 to rank 7 but remains outside top 5.

Next failure types:

- Topic-less profiles still over-match broad housing, family, and care services.
- Pregnancy-related cases need a way to distinguish health/benefit intent from
  broad family housing services.
- Low-income theme-specific cases such as energy, culture, medical, and finance
  need an explicit user purpose or a separate theme-intent signal.
- Child health and education cases need better separation between child-care,
  child-protection, and school/medical support.

## Goal 5 Evaluation Contract Audit

The old `expected_ids` field treated every listed service as the same kind of
answer. That was risky because the list mixed several meanings:

- strong matches supported by age, income, disability, pregnancy, child, or
  employment fields in `SearchRequest`
- broad acceptable alternatives that are reasonable in top results but are not
  guaranteed answers
- services that require extra facts not present in the request, such as high-risk
  pregnancy, actual childbirth, independent housing status, school status, or
  disease-specific eligibility
- false-positive guardrail services that should not appear near the top
- ambiguous profile-only cases where a single positive answer cannot be audited
  from metadata alone

The evaluation contract now uses `ServiceExpectation` for every positive and
negative service:

```python
ServiceExpectation(
    serv_id="WLF00001164",
    reason="67세 저소득 사용자는 기초연금의 노인 소득지원 대상성과 직접 맞닿는다.",
    evidence_field="tgtr_dtl_cn",
    condition=None,
)
```

Contract buckets:

- `must_ids`: strong answers supported by the request fields and indexed metadata.
- `acceptable_ids`: reasonable top-result candidates that should count as useful
  but are not strong enough to require as must hits.
- `conditional_ids`: candidates that need additional user facts before they can
  be treated as correct.
- `excluded_ids`: services that should not appear in top-5 for the explicit or
  missing conditions in the request.
- `ambiguous`: cases excluded from positive-hit denominators because the current
  profile is underspecified.

Examples:

- Must: `노인_저소득` marks `WLF00001164` 기초연금 as must because age 67
  and low-income status are directly supported by target metadata.
- Acceptable: theme-adjacent low-income housing, finance, education, culture, or
  employment services remain useful candidates without being forced into must.
- Conditional: `저소득_임산부` keeps high-risk pregnancy and emergency childbirth
  benefits as conditional because the request does not say high-risk pregnancy,
  actual childbirth, or emergency support status.
- Excluded: under-65 non-disabled or no-children profiles keep senior, disability,
  pregnancy, and parenting-only services as top-5 regressions.
- Ambiguous: generic "not 북한이탈/다문화/농어업" guardrail profiles are marked
  ambiguous because their positive answer cannot be uniquely derived from the
  available fields.

Metric definitions:

- `must_hit@1/3/5`: denominator is non-ambiguous cases with `must_ids`.
- `acceptable_hit@5`: denominator is non-ambiguous cases with `acceptable_ids`.
- `must_or_acceptable_hit@5`: denominator is non-ambiguous cases with either
  must or acceptable expectations.
- `mrr_must`: denominator is non-ambiguous cases with `must_ids`.
- `exclusion_pass@5`: denominator is all cases, including ambiguous cases.
- `exclusion_pass@5_on_excluded_cases`: denominator is only cases with
  `excluded_ids`.
- `conditional_hit@5`: diagnostic only; denominator is cases with
  `conditional_ids`.
- `exclusion_violations@5`: case-level list of excluded service IDs that appeared
  in the top-5.

Ambiguous and acceptable-only handling:

- `ambiguous=True` requires `ambiguity_reason` and removes the case from
  positive-hit and MRR denominators.
- Excluded checks still run for ambiguous cases.
- Acceptable-only cases are allowed. They contribute to `acceptable_hit@5` and
  `must_or_acceptable_hit@5`, but not to `must_hit@k` or `mrr_must`.
- A case with no must or acceptable services must be explicitly ambiguous or an
  excluded-only regression case.

Collision audit:

- Raw request collision reports group cases that share the same `SearchRequest`.
- Effective query collision reports group by `SearchRequest + query +
  intent_theme`.
- Theme-only cases such as culture, finance, medical, housing, care, education,
  and maternity require evaluation-only query text or intent theme. Without that
  separation, the same raw profile would be asked to satisfy mutually different
  answer contracts.

Remaining metadata risk:

- Some services mention broad target groups but depend on details that are not in
  `SearchRequest`, such as student status, household composition details,
  disease, employment insurance history, housing tenure, institutional status, or
  local program availability.
- Institution, business, farmer/fisher, veteran, defector, multicultural, and
  adoption/foster-care programs must not be promoted to personal-user `must_ids`
  unless the request explicitly contains that group. With the current production
  schema, those are usually excluded or conditional/acceptable at most.
- The audit intentionally does not change production ranking, section rerank, or
  negative penalty logic. Its purpose is to make the evaluation contract
  inspectable before more ranking tuning.

Latest contract size:

- Cases: 100
- Must expectations: 9
- Acceptable expectations: 530
- Conditional expectations: 3
- Excluded expectations: 127
- Ambiguous cases: 3

Latest Goal 5 metrics:

| Metric | Baseline | Section-aware |
| --- | ---: | ---: |
| `must_hit@1` | 0.375 (3/8) | 0.375 (3/8) |
| `must_hit@3` | 0.625 (5/8) | 0.625 (5/8) |
| `must_hit@5` | 0.625 (5/8) | 0.625 (5/8) |
| `acceptable_hit@5` | 0.773 (75/97) | 0.763 (74/97) |
| `must_or_acceptable_hit@5` | 0.784 (76/97) | 0.773 (75/97) |
| `exclusion_pass@5` | 0.990 (99/100) | 1.000 (100/100) |
| `exclusion_pass@5_on_excluded_cases` | 0.955 (21/22) | 1.000 (22/22) |
| `conditional_hit@5` | 0.000 (0/1) | 1.000 (1/1) |
| `mrr_must` | 0.500 | 0.500 |
| `exclusion_violations@5` | 1 | 0 |
| Avg latency | 48.3 ms | 65.8 ms |
| P95 latency | 54.3 ms | 77.5 ms |
| `likely` distribution | 788 / 1000 (0.788) | 777 / 1000 (0.777) |
| `needs_more_info` distribution | 212 / 1000 (0.212) | 223 / 1000 (0.223) |
| `unlikely` distribution | 0 / 1000 (0.000) | 0 / 1000 (0.000) |

Collision audit:

- Raw request collision groups: 10. These are expected profile duplicates that
  express different evaluation themes.
- Effective query collision groups: 0 after adding evaluation-only query/theme
  resolution.
- Baseline `exclusion_violations@5`: `61세_일반_노인전용오탐방지` returned
  `WLF00001155`.
- Section-aware `exclusion_violations@5`: none.

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

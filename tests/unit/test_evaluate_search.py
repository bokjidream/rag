from __future__ import annotations

from scripts.evaluate_search import (
    EvalCase,
    EvaluationRow,
    ServiceExpectation,
    _rank_of,
    _validate_eval_cases,
    compute_metrics,
    effective_query_collision_reports,
    raw_request_collision_reports,
    svc,
)
from src.models.welfare import SearchRequest


def _case(
    name: str,
    *,
    must_ids: tuple[ServiceExpectation, ...] = (),
    acceptable_ids: tuple[ServiceExpectation, ...] = (),
    conditional_ids: tuple[ServiceExpectation, ...] = (),
    excluded_ids: tuple[ServiceExpectation, ...] = (),
    ambiguous: bool = False,
    query: str | None = None,
) -> EvalCase:
    return EvalCase(
        name=name,
        request=SearchRequest(age=30, income_level="일반", top_k=10),
        acceptable_ids=acceptable_ids,
        notes="unit test case",
        excluded_ids=excluded_ids,
        must_ids=must_ids,
        conditional_ids=conditional_ids,
        ambiguous=ambiguous,
        ambiguity_reason="ambiguous unit test" if ambiguous else None,
        query=query,
    )


def _row(case: EvalCase, result_ids: tuple[str, ...]) -> EvaluationRow:
    excluded_ids = {expectation.serv_id for expectation in case.excluded_ids}
    return EvaluationRow(
        case=case,
        result_ids=result_ids,
        results=tuple((serv_id, serv_id, 1.0) for serv_id in result_ids),
        must_rank=_rank_of(result_ids, case.must_ids),
        acceptable_rank=_rank_of(result_ids, case.acceptable_ids),
        conditional_rank=_rank_of(result_ids, case.conditional_ids),
        excluded_hits_at5=tuple(serv_id for serv_id in result_ids[:5] if serv_id in excluded_ids),
    )


def _metadata(*serv_ids: str) -> dict[str, dict[str, str]]:
    return {
        serv_id: {
            "serv_nm": serv_id,
            "serv_dgst": "summary",
            "tgtr_dtl_cn": "target",
            "slct_crit_cn": "criteria",
            "trgter_indvdl": "[]",
            "intrs_thema": "[]",
        }
        for serv_id in serv_ids
    }


def test_must_hit_at_k_and_mrr_must() -> None:
    case = _case("must", must_ids=(svc("A", "must reason", "serv_nm"),))

    metrics = compute_metrics([_row(case, ("B", "A", "C"))])

    assert metrics.must_hit_at[1].numerator == 0
    assert metrics.must_hit_at[3].numerator == 1
    assert metrics.must_hit_at[5].numerator == 1
    assert metrics.must_hit_at[5].denominator == 1
    assert metrics.mrr_must.value == 0.5


def test_acceptable_only_is_excluded_from_must_denominator() -> None:
    case = _case("acceptable", acceptable_ids=(svc("A", "acceptable reason", "serv_nm"),))

    metrics = compute_metrics([_row(case, ("A", "B"))])

    assert metrics.must_hit_at[5].denominator == 0
    assert metrics.mrr_must.denominator == 0
    assert metrics.acceptable_hit_at5.numerator == 1
    assert metrics.acceptable_hit_at5.denominator == 1
    assert metrics.must_or_acceptable_hit_at5.numerator == 1
    assert metrics.must_or_acceptable_hit_at5.denominator == 1


def test_ambiguous_cases_leave_hit_denominators_but_keep_exclusion_checks() -> None:
    case = _case(
        "ambiguous",
        must_ids=(svc("A", "must reason", "serv_nm"),),
        excluded_ids=(svc("X", "excluded reason", "serv_nm"),),
        ambiguous=True,
    )

    metrics = compute_metrics([_row(case, ("X", "A"))])

    assert metrics.must_hit_at[5].denominator == 0
    assert metrics.must_or_acceptable_hit_at5.denominator == 0
    assert metrics.exclusion_pass_at5.numerator == 0
    assert metrics.exclusion_pass_at5.denominator == 1
    assert metrics.exclusion_violations_at5 == (("ambiguous", ("X",)),)


def test_exclusion_pass_on_excluded_cases_uses_only_cases_with_excluded_ids() -> None:
    excluded_case = _case(
        "excluded",
        acceptable_ids=(svc("A", "acceptable reason", "serv_nm"),),
        excluded_ids=(svc("X", "excluded reason", "serv_nm"),),
    )
    regular_case = _case("regular", acceptable_ids=(svc("B", "acceptable reason", "serv_nm"),))

    metrics = compute_metrics(
        [
            _row(excluded_case, ("X", "A")),
            _row(regular_case, ("B", "C")),
        ]
    )

    assert metrics.exclusion_pass_at5.numerator == 1
    assert metrics.exclusion_pass_at5.denominator == 2
    assert metrics.exclusion_pass_at5_on_excluded_cases.numerator == 0
    assert metrics.exclusion_pass_at5_on_excluded_cases.denominator == 1


def test_raw_and_effective_collision_detection() -> None:
    housing = _case(
        "housing",
        acceptable_ids=(svc("A", "acceptable reason", "serv_nm"),),
        query="주거",
    )
    medical = _case(
        "medical",
        acceptable_ids=(svc("B", "acceptable reason", "serv_nm"),),
        query="의료",
    )

    assert len(raw_request_collision_reports((housing, medical))) == 1
    assert effective_query_collision_reports((housing, medical)) == ()


def test_effective_collision_detects_excluded_positive_overlap() -> None:
    positive = _case(
        "positive",
        acceptable_ids=(svc("A", "acceptable reason", "serv_nm"),),
        query="same",
    )
    negative = _case(
        "negative",
        excluded_ids=(svc("A", "excluded reason", "serv_nm"),),
        query="same",
    )

    reports = effective_query_collision_reports((positive, negative))

    assert len(reports) == 1
    assert reports[0].conflict is True
    assert "overlaps" in reports[0].reason


def test_validation_requires_reason() -> None:
    case = _case(
        "missing_reason",
        acceptable_ids=(ServiceExpectation("A", "", "serv_nm"),),
    )

    errors = _validate_eval_cases(_metadata("A"), (case,), expected_case_count=None)

    assert any("reason is required" in error for error in errors)


def test_validation_rejects_duplicate_id_across_contract_buckets() -> None:
    case = _case(
        "duplicate",
        acceptable_ids=(svc("A", "acceptable reason", "serv_nm"),),
        excluded_ids=(svc("A", "excluded reason", "serv_nm"),),
    )

    errors = _validate_eval_cases(_metadata("A"), (case,), expected_case_count=None)

    assert any("also appears in acceptable_ids" in error for error in errors)

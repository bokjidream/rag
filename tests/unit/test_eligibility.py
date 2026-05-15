from __future__ import annotations

from src.models.welfare import SearchRequest
from src.retriever.eligibility import evaluate_eligibility


def _request(**overrides: object) -> SearchRequest:
    payload: dict[str, object] = {
        "age": 61,
        "income_level": "일반",
        "has_children": None,
        "disability": False,
        "employment_status": None,
    }
    payload.update(overrides)
    return SearchRequest(**payload)  # type: ignore[arg-type]


def test_age_under_65_with_exact_senior_criteria_is_unlikely() -> None:
    result = evaluate_eligibility(
        _request(age=61, disability=False),
        {
            "slct_crit_cn": "독거노인은 실제로 혼자 살고있는 만 65세 이상의 노인입니다.",
        },
    )

    assert result.status == "unlikely"
    assert any("만 65세 이상" in reason for reason in result.reasons)
    assert result.evidence[0].field == "slct_crit_cn"
    assert "만 65세 이상의 노인" in result.evidence[0].text


def test_age_guardrail_does_not_fire_on_under_65_phrase() -> None:
    result = evaluate_eligibility(
        _request(age=61),
        {
            "tgtr_dtl_cn": "만 65세 미만의 기준중위소득 70% 이하 계층을 지원합니다.",
        },
    )

    assert result.status == "likely"
    assert result.reasons == []


def test_disability_false_with_core_disability_criteria_is_unlikely() -> None:
    result = evaluate_eligibility(
        _request(age=40, disability=False),
        {
            "slct_crit_cn": (
                "장애인 가구는 장애인활동지원 수급자이면서 독거 또는 취약가구에 "
                "해당하는 장애인입니다."
            ),
        },
    )

    assert result.status == "unlikely"
    assert any("장애인" in reason for reason in result.reasons)


def test_disability_false_with_disability_as_one_of_many_conditions_needs_more_info() -> None:
    result = evaluate_eligibility(
        _request(age=40, disability=False),
        {
            "slct_crit_cn": (
                "장애의 정도가 심한 장애인, 6개월 이상 치료를 요하는 중증질환자, "
                "희귀난치성 질환자, 한부모가정 등이 신청할 수 있습니다."
            ),
        },
    )

    assert result.status == "needs_more_info"
    assert any("여러 조건" in reason for reason in result.reasons)


def test_strongest_status_wins_when_multiple_guardrails_fire() -> None:
    result = evaluate_eligibility(
        _request(age=61, has_children=False),
        {
            "slct_crit_cn": (
                "실제로 혼자 살고있는 만 65세 이상의 노인입니다. "
                "자녀장려금은 부양자녀가 있는 경우 적용됩니다."
            ),
        },
    )

    assert result.status == "unlikely"
    assert any("자녀" in reason for reason in result.reasons)
    assert any("만 65세 이상" in reason for reason in result.reasons)


def test_has_children_false_with_child_centered_criteria_needs_more_info() -> None:
    result = evaluate_eligibility(
        _request(has_children=False),
        {
            "slct_crit_cn": "자녀장려금은 부양자녀가 있는 경우 적용됩니다.",
        },
    )

    assert result.status == "needs_more_info"
    assert any("자녀" in reason for reason in result.reasons)


def test_has_children_none_does_not_apply_child_guardrail() -> None:
    result = evaluate_eligibility(
        _request(has_children=None),
        {
            "slct_crit_cn": "자녀장려금은 부양자녀가 있는 경우 적용됩니다.",
        },
    )

    assert result.status == "likely"


def test_non_economic_activity_with_income_or_self_support_criteria_needs_more_info() -> None:
    result = evaluate_eligibility(
        _request(employment_status="비경제활동"),
        {
            "slct_crit_cn": (
                "근로소득 또는 사업소득이 있는 거주자로서 요건을 갖추거나 "
                "자활사업 참여 자격이 있는 사람을 대상으로 합니다."
            ),
        },
    )

    assert result.status == "needs_more_info"
    assert any("근로" in reason or "자활" in reason for reason in result.reasons)


def test_low_income_with_only_basic_or_near_poverty_criteria_needs_more_info() -> None:
    result = evaluate_eligibility(
        _request(income_level="저소득"),
        {
            "tgtr_dtl_cn": "기초생활수급자 또는 차상위계층만 신청 가능합니다.",
        },
    )

    assert result.status == "needs_more_info"
    assert any("기초생활수급자" in reason for reason in result.reasons)


def test_missing_metadata_fields_are_handled_without_key_error() -> None:
    result = evaluate_eligibility(
        _request(),
        {
            "serv_dgst": "일반 복지 서비스입니다.",
        },
    )

    assert result.status == "likely"
    assert "tgtr_dtl_cn" in result.missing_fields
    assert "slct_crit_cn" in result.missing_fields
    assert "alw_serv_cn" in result.missing_fields


def test_likely_when_no_clear_conflict() -> None:
    result = evaluate_eligibility(
        _request(age=45, income_level="일반", employment_status="취업"),
        {
            "serv_dgst": "일반 가구의 생활 안정을 지원합니다.",
            "tgtr_dtl_cn": "지역 주민 누구나 신청할 수 있습니다.",
            "slct_crit_cn": "별도 선정 기준 없이 예산 범위에서 지원합니다.",
            "alw_serv_cn": "상담 및 서비스 연계를 제공합니다.",
        },
    )

    assert result.status == "likely"
    assert result.reasons == []
    assert result.evidence == []

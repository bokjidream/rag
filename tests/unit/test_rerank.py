from __future__ import annotations

import json

import pytest

from src.models.welfare import SearchRequest
from src.retriever.intent import build_query_intent
from src.retriever.rerank import negative_condition_penalty, rank_service_candidates


def _metadata(**overrides: str) -> dict[str, str]:
    metadata = {
        "serv_id": "SVC",
        "serv_nm": "일반 생활지원",
        "serv_dgst": "지역 주민을 위한 생활지원",
        "jur_mnof_nm": "보건복지부",
        "trgter_indvdl": json.dumps([], ensure_ascii=False),
        "intrs_thema": json.dumps(["생활지원"], ensure_ascii=False),
        "tgtr_dtl_cn": "지역 주민",
        "slct_crit_cn": "",
        "alw_serv_cn": "상담 지원",
    }
    metadata.update(overrides)
    return metadata


def test_negative_penalty_demotes_child_service_only_when_no_children_is_explicit() -> None:
    child_service = _metadata(
        serv_nm="한부모가족 아동양육비 지원",
        trgter_indvdl=json.dumps(["한부모·조손"], ensure_ascii=False),
        tgtr_dtl_cn="저소득 한부모가구의 18세 미만 아동",
    )
    no_children = build_query_intent(SearchRequest(age=27, income_level="일반", has_children=False))
    unknown_children = build_query_intent(SearchRequest(age=27, income_level="일반"))

    penalty, reasons = negative_condition_penalty(no_children, child_service)
    unknown_penalty, unknown_reasons = negative_condition_penalty(
        unknown_children,
        child_service,
    )

    assert penalty > 0
    assert "no_children" in reasons
    assert unknown_penalty == 0
    assert unknown_reasons == ()


def test_section_rerank_uses_weighted_sections_and_soft_penalty() -> None:
    request = SearchRequest(age=61, income_level="일반", has_children=False)
    intent = build_query_intent(request)
    senior_target = _metadata(
        serv_nm="노인일자리 및 사회활동 지원사업",
        serv_dgst="노인의 사회활동을 지원합니다.",
        tgtr_dtl_cn="사회활동 지원사업 참여 조건에 부합하는 65세 이상 어르신을 지원합니다.",
        chunk_section="target",
    )
    senior_application = {**senior_target, "chunk_section": "application"}

    ranked = rank_service_candidates(
        request,
        intent,
        [(senior_application, 0.05), (senior_target, 0.24)],
        enable_section_rerank=True,
    )

    assert ranked.section_scores.keys() == {"application", "target"}
    assert ranked.section_weighted_score < 1.0
    assert ranked.negative_penalty > 0
    assert "under_senior_age" in ranked.reasons
    assert ranked.score < 0.95


def test_section_rerank_disabled_keeps_best_chunk_score() -> None:
    request = SearchRequest(age=61, income_level="일반", has_children=False)
    intent = build_query_intent(request)
    target = _metadata(chunk_section="target")
    document = _metadata(chunk_section="documents")

    ranked = rank_service_candidates(
        request,
        intent,
        [(target, 0.30), (document, 0.05)],
        enable_section_rerank=False,
    )

    assert ranked.score == pytest.approx(0.95)
    assert ranked.negative_penalty == 0


def test_negative_penalty_does_not_demote_broad_service_with_incidental_group_terms() -> None:
    broad_service = _metadata(
        serv_nm="자활근로",
        serv_dgst="근로 능력이 있는 저소득층에게 일자리를 제공합니다.",
        trgter_indvdl=json.dumps(["저소득"], ensure_ascii=False),
        tgtr_dtl_cn="수급자, 차상위자, 다문화가족 등 여러 취약계층을 지원합니다.",
    )
    intent = build_query_intent(SearchRequest(age=50, income_level="차상위계층"))

    penalty, reasons = negative_condition_penalty(intent, broad_service)

    assert penalty == 0
    assert reasons == ()


def test_negative_penalty_preserves_not_disabled_and_not_pregnant_rules() -> None:
    disability_service = _metadata(
        serv_nm="장애수당",
        serv_dgst="저소득 장애인에게 생활 안정을 위한 수당을 지급합니다.",
        trgter_indvdl=json.dumps(["장애인"], ensure_ascii=False),
    )
    pregnancy_service = _metadata(
        serv_nm="표준모자보건수첩 제공",
        serv_dgst="임산부에게 임신과 출산 관련 건강관리 정보를 제공합니다.",
        trgter_indvdl=json.dumps(["임신·출산"], ensure_ascii=False),
    )
    intent = build_query_intent(SearchRequest(age=30, income_level="일반", pregnant=False))

    disability_penalty, disability_reasons = negative_condition_penalty(
        intent,
        disability_service,
    )
    pregnancy_penalty, pregnancy_reasons = negative_condition_penalty(
        intent,
        pregnancy_service,
    )

    assert disability_penalty > 0
    assert "not_disabled" in disability_reasons
    assert pregnancy_penalty > 0
    assert "not_pregnant" in pregnancy_reasons


def test_child_care_theme_lifts_general_care_above_special_parenting_targets() -> None:
    request = SearchRequest(age=5, income_level="일반")
    intent = build_query_intent(request, query_text="5세 영유아 아동 보육")
    care_service = _metadata(
        serv_nm="가정양육수당 지원사업",
        serv_dgst="어린이집 등을 이용하지 않는 영유아에게 양육수당을 지원합니다.",
        tgtr_dtl_cn="취학 전 영유아",
        chunk_section="target",
    )
    special_parenting = _metadata(
        serv_nm="한부모가족 아동양육비 지원",
        serv_dgst="저소득 한부모가족의 아동양육비를 지원합니다.",
        trgter_indvdl=json.dumps(["한부모·조손"], ensure_ascii=False),
        tgtr_dtl_cn="한부모가족의 18세 미만 아동",
        chunk_section="target",
    )

    care_ranked = rank_service_candidates(
        request,
        intent,
        [(care_service, 0.20)],
        enable_section_rerank=True,
    )
    special_ranked = rank_service_candidates(
        request,
        intent,
        [(special_parenting, 0.20)],
        enable_section_rerank=True,
    )

    assert care_ranked.theme_adjustment > 0
    assert special_ranked.theme_adjustment < 0
    assert care_ranked.score > special_ranked.score


def test_child_support_theme_keeps_parenting_support_from_child_penalty() -> None:
    request = SearchRequest(
        age=36,
        income_level="저소득",
        marital_status="이혼",
        has_children=True,
    )
    intent = build_query_intent(
        request,
        query_text="한부모 양육비 자녀 양육 양육비 이행",
        intent_theme="child-support",
    )
    child_support = _metadata(
        serv_nm="한부모가족 아동양육비 지원",
        serv_dgst="저소득 한부모가족의 아동양육비를 지원합니다.",
        trgter_indvdl=json.dumps(["한부모·조손"], ensure_ascii=False),
        tgtr_dtl_cn="한부모가족의 18세 미만 아동",
        chunk_section="target",
    )

    ranked = rank_service_candidates(
        request,
        intent,
        [(child_support, 0.20)],
        enable_section_rerank=True,
    )

    assert "theme:child_penalty" not in ranked.reasons


def test_maternity_theme_lifts_birth_service_above_general_housing() -> None:
    request = SearchRequest(age=30, income_level="저소득", pregnant=True)
    intent = build_query_intent(
        request,
        query_text="임신 출산 산모 건강관리 모자보건",
        intent_theme="maternity",
    )
    maternity_service = _metadata(
        serv_nm="표준모자보건수첩 제공",
        serv_dgst="임산부에게 임신과 출산 관련 건강관리 정보를 제공합니다.",
        trgter_indvdl=json.dumps(["임신·출산"], ensure_ascii=False),
        tgtr_dtl_cn="임산부",
        chunk_section="target",
    )
    housing_service = _metadata(
        serv_nm="기존주택 전세임대주택 지원사업",
        serv_dgst="저소득층에게 전세임대 주택을 지원합니다.",
        intrs_thema=json.dumps(["주거"], ensure_ascii=False),
        tgtr_dtl_cn="저소득 무주택 세대",
        chunk_section="target",
    )

    maternity_ranked = rank_service_candidates(
        request,
        intent,
        [(maternity_service, 0.20)],
        enable_section_rerank=True,
    )
    housing_ranked = rank_service_candidates(
        request,
        intent,
        [(housing_service, 0.20)],
        enable_section_rerank=True,
    )

    assert maternity_ranked.theme_adjustment > 0
    assert housing_ranked.theme_adjustment < 0
    assert maternity_ranked.score > housing_ranked.score


def test_low_income_theme_penalizes_broad_housing_mismatch_but_not_housing_theme() -> None:
    request = SearchRequest(age=30, income_level="저소득")
    culture_intent = build_query_intent(
        request,
        query_text="문화 여가 스포츠 교육 바우처 이용권",
        intent_theme="culture",
    )
    housing_intent = build_query_intent(
        request,
        query_text="주거 주거급여 공공임대 전세 월세 주거비",
        intent_theme="housing",
    )
    housing_service = _metadata(
        serv_nm="기존주택 전세임대주택 지원사업",
        serv_dgst="저소득층에게 전세임대 주택을 지원합니다.",
        intrs_thema=json.dumps(["주거"], ensure_ascii=False),
        tgtr_dtl_cn="저소득 무주택 세대",
        chunk_section="target",
    )

    culture_ranked = rank_service_candidates(
        request,
        culture_intent,
        [(housing_service, 0.20)],
        enable_section_rerank=True,
    )
    housing_ranked = rank_service_candidates(
        request,
        housing_intent,
        [(housing_service, 0.20)],
        enable_section_rerank=True,
    )

    assert culture_ranked.theme_adjustment < 0
    assert housing_ranked.theme_adjustment > 0
    assert "theme:housing_boost" in housing_ranked.reasons


def test_startup_theme_lifts_business_financing_above_generic_disability_support() -> None:
    request = SearchRequest(
        age=45,
        income_level="저소득",
        disability=True,
        disability_severity="경증",
        employment_status="실업",
    )
    intent = build_query_intent(
        request,
        query_text="창업 점포 기업 인턴 고용 융자",
        intent_theme="startup",
    )
    startup_service = _metadata(
        serv_nm="장애인 창업점포 지원사업",
        serv_dgst="장애인 창업을 위한 점포와 사업화를 지원합니다.",
        trgter_indvdl=json.dumps(["장애인"], ensure_ascii=False),
        tgtr_dtl_cn="창업을 희망하는 장애인",
        chunk_section="target",
    )
    generic_service = _metadata(
        serv_nm="장애수당",
        serv_dgst="저소득 장애인에게 생활 안정을 위한 수당을 지급합니다.",
        trgter_indvdl=json.dumps(["장애인"], ensure_ascii=False),
        tgtr_dtl_cn="등록 장애인",
        chunk_section="target",
    )

    startup_ranked = rank_service_candidates(
        request,
        intent,
        [(startup_service, 0.20)],
        enable_section_rerank=True,
    )
    generic_ranked = rank_service_candidates(
        request,
        intent,
        [(generic_service, 0.20)],
        enable_section_rerank=True,
    )

    assert startup_ranked.theme_adjustment > 0
    assert generic_ranked.theme_adjustment < 0
    assert startup_ranked.score > generic_ranked.score


def test_guardrail_theme_does_not_apply_theme_adjustment() -> None:
    request = SearchRequest(age=30, income_level="일반", disability=False)
    intent = build_query_intent(
        request,
        query_text="30세 일반 거주자를 위한 복지 서비스",
        intent_theme="guardrail:not_disabled",
    )
    disability_service = _metadata(
        serv_nm="장애수당",
        serv_dgst="저소득 장애인에게 생활 안정을 위한 수당을 지급합니다.",
        trgter_indvdl=json.dumps(["장애인"], ensure_ascii=False),
        tgtr_dtl_cn="등록 장애인",
        chunk_section="target",
    )

    ranked = rank_service_candidates(
        request,
        intent,
        [(disability_service, 0.20)],
        enable_section_rerank=True,
    )

    assert ranked.theme_adjustment == 0
    assert all(not reason.startswith("theme:") for reason in ranked.reasons)


def test_section_rerank_disabled_ignores_theme_adjustment() -> None:
    request = SearchRequest(age=30, income_level="일반", pregnant=False)
    intent = build_query_intent(
        request,
        query_text="임신 출산 산모 건강관리 모자보건",
        intent_theme="maternity",
    )
    maternity_service = _metadata(
        serv_nm="표준모자보건수첩 제공",
        serv_dgst="임산부에게 임신과 출산 관련 건강관리 정보를 제공합니다.",
        tgtr_dtl_cn="임산부",
        chunk_section="target",
    )

    ranked = rank_service_candidates(
        request,
        intent,
        [(maternity_service, 0.30)],
        enable_section_rerank=False,
    )

    assert ranked.theme_adjustment == 0
    assert ranked.score == pytest.approx(0.70)

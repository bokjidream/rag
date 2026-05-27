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
    no_children = build_query_intent(
        SearchRequest(age=27, income_level="일반", has_children=False)
    )
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

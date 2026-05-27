from __future__ import annotations

from src.models.welfare import SearchRequest
from src.retriever.intent import build_query_intent


def test_build_query_intent_distinguishes_no_children_from_unknown() -> None:
    no_children = build_query_intent(
        SearchRequest(age=30, income_level="일반", has_children=False)
    )
    unknown_children = build_query_intent(SearchRequest(age=30, income_level="일반"))

    assert "no_children" in no_children.negative_flags
    assert "children" not in no_children.unknown_flags
    assert "no_children" not in unknown_children.negative_flags
    assert "children" in unknown_children.unknown_flags


def test_build_query_intent_marks_unrepresented_group_conditions_unknown() -> None:
    intent = build_query_intent(SearchRequest(age=42, income_level="일반"))

    assert {
        "veteran",
        "agriculture_or_fishery",
        "multicultural",
        "north_korean_defector",
    } <= intent.unknown_flags

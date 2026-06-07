from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.models.welfare import SearchRequest

AgeGroup = Literal["child", "youth", "young_adult", "adult", "senior"]


@dataclass(frozen=True)
class QueryIntent:
    age_group: AgeGroup
    income_level: str
    has_disability: bool
    has_pregnancy: bool
    has_children: bool | None
    negative_flags: frozenset[str]
    unknown_flags: frozenset[str]
    intent_theme: str | None = None
    query_text: str | None = None


def build_query_intent(
    request: SearchRequest,
    *,
    query_text: str | None = None,
    intent_theme: str | None = None,
) -> QueryIntent:
    """SearchRequest를 reranking rule 입력으로 정규화한다."""
    negative_flags: set[str] = set()
    unknown_flags: set[str] = {
        "veteran",
        "agriculture_or_fishery",
        "multicultural",
        "north_korean_defector",
    }

    if request.age < 65:
        negative_flags.add("under_senior_age")
    if not request.disability:
        negative_flags.add("not_disabled")
    if not request.pregnant:
        negative_flags.add("not_pregnant")
    if request.has_children is False:
        negative_flags.add("no_children")
    elif request.has_children is None:
        unknown_flags.add("children")

    return QueryIntent(
        age_group=_age_group(request.age),
        income_level=request.income_level,
        has_disability=request.disability,
        has_pregnancy=request.pregnant,
        has_children=request.has_children,
        negative_flags=frozenset(negative_flags),
        unknown_flags=frozenset(unknown_flags),
        intent_theme=intent_theme,
        query_text=query_text,
    )


def _age_group(age: int) -> AgeGroup:
    if age <= 12:
        return "child"
    if age <= 18:
        return "youth"
    if age <= 34:
        return "young_adult"
    if age < 65:
        return "adult"
    return "senior"

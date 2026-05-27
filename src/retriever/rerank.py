from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.models.welfare import SearchRequest
from src.retriever.intent import QueryIntent

DOCUMENT_SECTION = "document"
PRIMARY_SECTIONS = {"target", "criteria"}
SECTION_WEIGHTS: dict[str, float] = {
    "target": 1.0,
    "criteria": 0.95,
    "benefit": 0.72,
    "summary": 0.58,
    "application": 0.28,
    "documents": 0.22,
    DOCUMENT_SECTION: 1.0,
}
SECTION_ADJUSTMENTS: dict[str, float] = {
    "target": 0.025,
    "criteria": 0.02,
    "benefit": 0.0,
    "summary": -0.005,
    "application": -0.06,
    "documents": -0.08,
    DOCUMENT_SECTION: 0.0,
}
PRIMARY_SECTION_BOOST = 0.015
MAX_PRIMARY_SECTION_BOOST = 0.04

_SENIOR_TERMS = ("노인", "어르신", "고령", "65세 이상", "만 65세 이상", "기초연금")
_DISABILITY_TERMS = (
    "장애인",
    "장애아",
    "발달장애",
    "중증장애",
    "시각장애",
    "청각장애",
)
_PREGNANCY_TERMS = ("임산부", "임신", "출산", "산모", "해산", "모자보건")
_CHILD_ONLY_TERMS = (
    "한부모",
    "조손",
    "청소년부모",
    "아동양육",
    "자녀 양육",
    "아이돌봄",
    "보육료",
    "아동수당",
)
_VETERAN_TERMS = ("보훈", "국가유공", "독립유공", "참전유공", "보훈대상")
_AGRICULTURE_TERMS = ("농업인", "어업인", "농어업", "농촌", "어촌", "축산업", "임업")
_MULTICULTURAL_TERMS = ("다문화", "결혼이민")
_NORTH_KOREAN_DEFECTOR_TERMS = ("북한이탈", "탈북민")


@dataclass(frozen=True)
class RankedService:
    metadata: dict[str, str]
    distance: float
    score: float
    raw_score: float
    profile_boost: float
    section_scores: dict[str, float] = field(default_factory=dict)
    section_weighted_score: float = 0.0
    section_evidence_boost: float = 0.0
    negative_penalty: float = 0.0
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ChunkScore:
    metadata: dict[str, str]
    distance: float
    section: str
    raw_score: float
    profile_boost: float
    score: float


def rank_service_candidates(
    request: SearchRequest,
    intent: QueryIntent,
    candidates: Sequence[tuple[dict[str, str], float]],
    *,
    enable_section_rerank: bool,
) -> RankedService:
    if not candidates:
        raise ValueError("candidates must not be empty")

    service_profile_boost = _profile_boost(request, candidates[0][0])
    chunk_scores = [
        _score_chunk(request, metadata, distance, profile_boost=service_profile_boost)
        for metadata, distance in candidates
    ]
    if not enable_section_rerank or not _has_section_chunks(chunk_scores):
        return _rank_by_best_chunk(chunk_scores)

    section_best = _best_by_section(chunk_scores)
    section_scores = {section: chunk.score for section, chunk in section_best.items()}
    adjusted_scores = {
        section: _clamp_score(score + SECTION_ADJUSTMENTS.get(section, 0.0))
        for section, score in section_scores.items()
    }
    weighted_score = _weighted_section_score(adjusted_scores)
    best_section, best_adjusted_score = max(adjusted_scores.items(), key=lambda item: item[1])
    best_chunk = section_best[best_section]
    evidence_boost = _primary_section_boost(section_scores)
    penalty, reasons = negative_condition_penalty(intent, best_chunk.metadata)
    final_score = _clamp_score(
        (best_adjusted_score * 0.72) + (weighted_score * 0.28) + evidence_boost - penalty
    )

    return RankedService(
        metadata=best_chunk.metadata,
        distance=best_chunk.distance,
        score=final_score,
        raw_score=best_chunk.raw_score,
        profile_boost=best_chunk.profile_boost,
        section_scores=section_scores,
        section_weighted_score=weighted_score,
        section_evidence_boost=evidence_boost,
        negative_penalty=penalty,
        reasons=reasons,
    )


def negative_condition_penalty(
    intent: QueryIntent,
    metadata: Mapping[str, object],
) -> tuple[float, tuple[str, ...]]:
    targets, _, text = _metadata_terms(metadata)
    target_set = set(targets)
    service_text = _service_text(metadata)
    penalty = 0.0
    reasons: list[str] = []

    if "under_senior_age" in intent.negative_flags and _is_senior_dedicated(
        target_set,
        service_text,
        text,
    ):
        penalty += 0.09
        reasons.append("under_senior_age")

    if "not_disabled" in intent.negative_flags and _is_disability_dedicated(
        target_set,
        service_text,
        text,
    ):
        penalty += 0.12
        reasons.append("not_disabled")

    if "not_pregnant" in intent.negative_flags and _is_pregnancy_dedicated(
        target_set,
        service_text,
    ):
        penalty += 0.12
        reasons.append("not_pregnant")

    if "no_children" in intent.negative_flags and _is_child_or_parenting_dedicated(
        target_set,
        service_text,
    ):
        penalty += 0.14
        reasons.append("no_children")

    if "veteran" in intent.unknown_flags and _is_unknown_group_dedicated(
        target_set,
        service_text,
        ("보훈대상자",),
        _VETERAN_TERMS,
    ):
        penalty += 0.07
        reasons.append("unknown_veteran")

    if "agriculture_or_fishery" in intent.unknown_flags and _contains_any(
        service_text,
        _AGRICULTURE_TERMS,
    ):
        penalty += 0.07
        reasons.append("unknown_agriculture_or_fishery")

    if "multicultural" in intent.unknown_flags and _is_unknown_group_dedicated(
        target_set,
        service_text,
        ("다문화·탈북민",),
        _MULTICULTURAL_TERMS,
    ):
        penalty += 0.07
        reasons.append("unknown_multicultural")

    if "north_korean_defector" in intent.unknown_flags and _is_unknown_group_dedicated(
        target_set,
        service_text,
        ("다문화·탈북민",),
        _NORTH_KOREAN_DEFECTOR_TERMS,
    ):
        penalty += 0.07
        reasons.append("unknown_north_korean_defector")

    return min(0.28, penalty), tuple(reasons)


def _score_chunk(
    request: SearchRequest,
    metadata: dict[str, str],
    distance: float,
    *,
    profile_boost: float | None = None,
) -> _ChunkScore:
    resolved_profile_boost = (
        _profile_boost(request, metadata) if profile_boost is None else profile_boost
    )
    raw_score = max(0.0, 1.0 - distance)
    return _ChunkScore(
        metadata=metadata,
        distance=distance,
        section=str(metadata.get("chunk_section", DOCUMENT_SECTION) or DOCUMENT_SECTION),
        raw_score=raw_score,
        profile_boost=resolved_profile_boost,
        score=_clamp_score(raw_score + resolved_profile_boost),
    )


def _rank_by_best_chunk(chunk_scores: Sequence[_ChunkScore]) -> RankedService:
    best_chunk = max(chunk_scores, key=lambda chunk: chunk.score)
    return RankedService(
        metadata=best_chunk.metadata,
        distance=best_chunk.distance,
        score=best_chunk.score,
        raw_score=best_chunk.raw_score,
        profile_boost=best_chunk.profile_boost,
        section_scores={best_chunk.section: best_chunk.score},
        section_weighted_score=best_chunk.score,
    )


def _has_section_chunks(chunk_scores: Sequence[_ChunkScore]) -> bool:
    return any(chunk.section != DOCUMENT_SECTION for chunk in chunk_scores)


def _best_by_section(chunk_scores: Sequence[_ChunkScore]) -> dict[str, _ChunkScore]:
    best: dict[str, _ChunkScore] = {}
    for chunk in chunk_scores:
        current = best.get(chunk.section)
        if current is None or chunk.score > current.score:
            best[chunk.section] = chunk
    return best


def _weighted_section_score(section_scores: Mapping[str, float]) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for section, score in section_scores.items():
        weight = SECTION_WEIGHTS.get(section, 0.5)
        weighted_sum += score * weight
        weight_sum += weight
    if weight_sum == 0:
        return 0.0
    return _clamp_score(weighted_sum / weight_sum)


def _primary_section_boost(section_scores: Mapping[str, float]) -> float:
    matched_primary_sections = [
        section
        for section in PRIMARY_SECTIONS
        if section_scores.get(section, 0.0) >= 0.55
    ]
    return min(MAX_PRIMARY_SECTION_BOOST, len(matched_primary_sections) * PRIMARY_SECTION_BOOST)


def _clamp_score(score: float) -> float:
    return min(1.0, max(0.0, score))


def _metadata_terms(metadata: Mapping[str, object]) -> tuple[list[str], list[str], str]:
    targets = _json_str_list(metadata.get("trgter_indvdl", "[]"))
    themes = _json_str_list(metadata.get("intrs_thema", "[]"))
    text = " ".join(
        str(metadata.get(field, ""))
        for field in (
            "serv_nm",
            "serv_dgst",
            "tgtr_dtl_cn",
            "slct_crit_cn",
            "alw_serv_cn",
        )
    )
    return targets, themes, text


def _service_text(metadata: Mapping[str, object]) -> str:
    return " ".join(str(metadata.get(field, "")) for field in ("serv_nm", "serv_dgst"))


def _json_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _profile_boost(request: SearchRequest, metadata: Mapping[str, object]) -> float:
    targets, themes, text = _metadata_terms(metadata)
    target_set = set(targets)
    theme_set = set(themes)
    boost = 0.0

    if request.age >= 65 and any(term in text for term in ["노인", "어르신", "고령"]):
        boost += 0.08
    elif 19 <= request.age <= 34 and "청년" in text:
        boost += 0.08
    elif request.age <= 18 and any(term in text for term in ["아동", "청소년", "초중고", "학교"]):
        boost += 0.08

    if request.income_level in {"기초생활수급자", "차상위계층", "저소득"}:
        if "저소득" in target_set:
            boost += 0.06
        if request.income_level == "기초생활수급자" and any(
            term in text for term in ["기초생활수급", "국민기초생활보장", "생계급여"]
        ):
            boost += 0.08
        if request.income_level == "차상위계층" and "차상위" in text:
            boost += 0.08

    if request.disability:
        if "장애인" in target_set or "장애" in text:
            boost += 0.12
    elif target_set == {"장애인"}:
        boost -= 0.04

    if request.has_children:
        if theme_set & {"보육", "교육", "보호·돌봄"}:
            boost += 0.05
        if any(term in text for term in ["아동", "양육", "보육", "아이돌봄", "교육비"]):
            boost += 0.05
        if request.marital_status in {"이혼", "사별"} and "한부모·조손" in target_set:
            boost += 0.10
        elif "한부모" in text or "청소년부모" in text:
            boost -= 0.08

    if request.pregnant and any(
        term in text for term in ["임산부", "임신", "출산", "산모", "해산"]
    ):
        boost += 0.15

    if request.employment_status == "실업":
        if "일자리" in theme_set:
            boost += 0.06
        if any(term in text for term in ["취업", "고용", "구직", "직업훈련", "자활"]):
            boost += 0.08

    if "보훈대상자" in target_set or any(term in text for term in ["보훈", "국가유공", "독립유공"]):
        boost -= 0.12
    if target_set == {"다문화·탈북민"}:
        boost -= 0.08

    return boost


def _rank_score(request: SearchRequest, metadata: Mapping[str, object], distance: float) -> float:
    base_score = max(0.0, 1.0 - distance)
    return _clamp_score(base_score + _profile_boost(request, metadata))


def _is_senior_dedicated(target_set: set[str], service_text: str, text: str) -> bool:
    if target_set and target_set <= {"노인"}:
        return True
    if _contains_any(service_text, ("노인", "어르신", "고령", "기초연금")):
        return True
    return _contains_any(text, _SENIOR_TERMS) and _contains_any(
        text,
        (
            "65세 이상 어르신을 지원",
            "만 65세 이상인 어르신",
            "65세 이상인 분",
            "65세 이상의 노인",
        ),
    )


def _is_disability_dedicated(target_set: set[str], service_text: str, text: str) -> bool:
    if target_set and target_set <= {"장애인"}:
        return True
    if _contains_any(service_text, _DISABILITY_TERMS):
        return True
    return _contains_any(text, _DISABILITY_TERMS) and _contains_any(
        text,
        ("등록 장애인", "중증 장애인", "장애인활동지원", "장애인연금", "장애수당"),
    )


def _is_pregnancy_dedicated(target_set: set[str], service_text: str) -> bool:
    if target_set and target_set <= {"임신·출산"}:
        return True
    return _contains_any(service_text, _PREGNANCY_TERMS) and not _contains_any(
        service_text,
        ("신혼부부", "다자녀", "한부모"),
    )


def _is_child_or_parenting_dedicated(target_set: set[str], text: str) -> bool:
    if target_set and target_set <= {"한부모·조손", "아동", "영유아", "다자녀"}:
        return True
    return _contains_any(text, _CHILD_ONLY_TERMS)


def _is_unknown_group_dedicated(
    target_set: set[str],
    text: str,
    target_terms: tuple[str, ...],
    text_terms: tuple[str, ...],
) -> bool:
    if target_set and target_set <= set(target_terms):
        return True
    return _contains_any(text, text_terms)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)

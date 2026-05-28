from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
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
_THEME_METADATA_FIELDS = (
    "serv_nm",
    "serv_dgst",
    "tgtr_dtl_cn",
    "slct_crit_cn",
    "alw_serv_cn",
)
_CHILD_CARE_BOOST_TERMS = (
    "가정양육수당",
    "영유아보육료",
    "보육료",
    "유아학비",
    "아동수당",
    "아이돌봄",
    "다함께 돌봄",
    "다함께돌봄",
    "예방접종",
    "영유아 건강검진",
    "영유아건강검진",
    "시간제보육",
    "육아종합지원",
)
_CHILD_CARE_PENALTY_TERMS = (
    "청소년부모",
    "자립준비청년",
    "보호종료아동",
    "입양",
    "위탁",
)
_CHILD_SUPPORT_TERMS = ("한부모", "양육비")
_MATERNITY_BOOST_TERMS = (
    "표준모자보건수첩",
    "산모·신생아",
    "산모 신생아",
    "임신",
    "출산",
    "진료비",
    "해산급여",
    "해산비",
    "영양플러스",
    "기저귀",
    "조제분유",
    "모자보건",
)
_MATERNITY_MISMATCH_TERMS = (
    "전세임대",
    "매입임대",
    "한부모가족복지시설",
    "온가족보듬",
    "아동통합서비스",
    "방과후보육료",
)
_STARTUP_STRONG_TERMS = ("창업", "점포", "기업", "융자", "자립자금", "인턴")
_STARTUP_WEAK_TERMS = ("고용",)
_STARTUP_GENERIC_DISABILITY_TERMS = (
    "장애수당",
    "장애인연금",
    "전세임대",
    "매입임대",
    "장애인일자리",
    "장애인활동지원",
    "장애인보조기기",
    "평생교육이용권",
    "출퇴근비용",
)
_LOW_INCOME_THEME_TERMS: dict[str, tuple[str, ...]] = {
    "culture": (
        "문화",
        "스포츠",
        "여가",
        "바우처",
        "이용권",
        "통합문화이용권",
        "강좌",
        "예술",
    ),
    "medical": (
        "의료급여",
        "의료비",
        "건강검진",
        "질환",
        "치료비",
        "검진",
        "예방접종",
    ),
    "finance": (
        "서민금융",
        "학자금",
        "대출",
        "신용",
        "부채",
        "자산형성",
        "저축계좌",
    ),
    "care": (
        "돌봄",
        "안부",
        "안전",
        "방문",
        "재가",
        "간병",
        "다함께돌봄",
        "아이돌봄",
    ),
    "housing": (
        "주거",
        "주거급여",
        "전세",
        "월세",
        "임대",
        "주택",
        "주거비",
    ),
    "basic-living": (
        "생계",
        "생활지원",
        "양곡",
        "감면",
        "긴급복지",
        "에너지",
        "교육급여",
        "자활",
        "가사·간병",
        "가사 간병",
    ),
}
_BROAD_SERVICE_TERMS = (
    "통합사례관리",
    "아동통합서비스",
    "온가족보듬",
    "지역사회서비스 투자사업",
)
_BROAD_HOUSING_TERMS = ("주거급여", "전세임대", "매입임대", "행복주택", "월세")
_BROAD_EMPLOYMENT_TERMS = ("국민취업지원", "고용", "취업", "일자리", "직업훈련")


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
    theme_adjustment: float = 0.0
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

    service_metadata = _service_level_metadata(candidates)
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
    theme_adjustment, theme_reasons = intent_theme_adjustment(intent, service_metadata)
    final_score = _clamp_score(
        (best_adjusted_score * 0.72)
        + (weighted_score * 0.28)
        + evidence_boost
        + theme_adjustment
        - penalty
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
        theme_adjustment=theme_adjustment,
        negative_penalty=penalty,
        reasons=(*reasons, *theme_reasons),
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


def intent_theme_adjustment(
    intent: QueryIntent,
    metadata: Mapping[str, object],
) -> tuple[float, tuple[str, ...]]:
    """Apply small query-theme nudges for section-aware reranking only."""
    theme = intent.intent_theme or ""
    if theme.startswith("guardrail:"):
        return 0.0, ()

    _, _, text = _metadata_terms(metadata)
    service_text = _service_theme_text(metadata)
    query_text = intent.query_text or ""
    adjustment = 0.0
    reasons: list[str] = []

    child_adjustment, child_reasons = _child_theme_adjustment(intent, service_text, query_text)
    adjustment += child_adjustment
    reasons.extend(child_reasons)

    maternity_adjustment, maternity_reasons = _maternity_theme_adjustment(
        intent,
        service_text,
    )
    adjustment += maternity_adjustment
    reasons.extend(maternity_reasons)

    startup_adjustment, startup_reasons = _startup_theme_adjustment(theme, service_text)
    adjustment += startup_adjustment
    reasons.extend(startup_reasons)

    low_income_adjustment, low_income_reasons = _low_income_theme_adjustment(
        intent,
        service_text,
        text,
    )
    adjustment += low_income_adjustment
    reasons.extend(low_income_reasons)

    return _clamp_theme_adjustment(adjustment), tuple(reasons)


def _child_theme_adjustment(
    intent: QueryIntent,
    service_text: str,
    query_text: str,
) -> tuple[float, tuple[str, ...]]:
    theme = intent.intent_theme or ""
    if not (
        intent.age_group == "child"
        or "영유아" in query_text
        or ("아동" in query_text and theme != "child-support")
    ):
        return 0.0, ()

    adjustment = 0.0
    reasons: list[str] = []
    boost = _term_boost(service_text, _CHILD_CARE_BOOST_TERMS, base=0.035, per=0.012, cap=0.08)
    if boost > 0:
        adjustment += boost
        reasons.append("theme:child_boost")

    penalty_terms = _CHILD_CARE_PENALTY_TERMS
    if theme != "child-support":
        penalty_terms = (*penalty_terms, *_CHILD_SUPPORT_TERMS)
    penalty = _term_penalty(service_text, penalty_terms, base=0.035, per=0.012, cap=0.075)
    if penalty > 0:
        adjustment -= penalty
        reasons.append("theme:child_penalty")

    if (
        intent.age_group == "child"
        and _contains_any(service_text, ("청소년",))
        and not _contains_any(
            service_text,
            ("아동", "영유아", "초등", "보육", "아동수당"),
        )
    ):
        adjustment -= 0.03
        reasons.append("theme:child_age_penalty")

    return adjustment, tuple(reasons)


def _maternity_theme_adjustment(
    intent: QueryIntent,
    service_text: str,
) -> tuple[float, tuple[str, ...]]:
    theme = intent.intent_theme or ""
    if not (intent.has_pregnancy or theme.startswith("maternity")):
        return 0.0, ()

    adjustment = 0.0
    reasons: list[str] = []
    boost = _term_boost(service_text, _MATERNITY_BOOST_TERMS, base=0.035, per=0.012, cap=0.085)
    if boost > 0:
        adjustment += boost
        reasons.append("theme:maternity_boost")

    penalty = _term_penalty(
        service_text,
        _MATERNITY_MISMATCH_TERMS,
        base=0.055,
        per=0.012,
        cap=0.09,
    )
    if penalty > 0:
        adjustment -= penalty
        reasons.append("theme:maternity_penalty")

    return adjustment, tuple(reasons)


def _startup_theme_adjustment(
    theme: str,
    service_text: str,
) -> tuple[float, tuple[str, ...]]:
    if theme != "startup":
        return 0.0, ()

    adjustment = 0.0
    reasons: list[str] = []
    strong_boost = _term_boost(
        service_text, _STARTUP_STRONG_TERMS, base=0.045, per=0.014, cap=0.085
    )
    if strong_boost > 0:
        adjustment += strong_boost
        if _contains_any(service_text, _STARTUP_WEAK_TERMS):
            adjustment += 0.01
        reasons.append("theme:startup_boost")

    if strong_boost == 0 and _contains_any(service_text, _STARTUP_GENERIC_DISABILITY_TERMS):
        adjustment -= 0.07
        reasons.append("theme:startup_generic_penalty")
    elif _contains_any(service_text, ("장애수당", "장애인연금", "전세임대", "매입임대")):
        adjustment -= 0.04
        reasons.append("theme:startup_weak_penalty")

    return adjustment, tuple(reasons)


def _low_income_theme_adjustment(
    intent: QueryIntent,
    service_text: str,
    metadata_text: str,
) -> tuple[float, tuple[str, ...]]:
    theme = intent.intent_theme or ""
    terms = _LOW_INCOME_THEME_TERMS.get(theme)
    if terms is None:
        return 0.0, ()

    adjustment = 0.0
    reasons: list[str] = []
    boost = _term_boost(service_text, terms, base=0.03, per=0.01, cap=0.075)
    if boost > 0:
        adjustment += boost
        reasons.append(f"theme:{theme}_boost")

    has_target_terms = boost > 0 or _contains_any(metadata_text, terms)
    mismatch_penalty = 0.0
    if not has_target_terms:
        other_terms = tuple(
            term
            for other_theme, other_theme_terms in _LOW_INCOME_THEME_TERMS.items()
            if other_theme != theme
            for term in other_theme_terms
        )
        mismatch_penalty += _term_penalty(service_text, other_terms, base=0.02, per=0.006, cap=0.04)

    if theme not in {"housing", "basic-living"} and not has_target_terms:
        mismatch_penalty += _term_penalty(
            service_text,
            _BROAD_HOUSING_TERMS,
            base=0.025,
            per=0.006,
            cap=0.045,
        )
    if theme != "employment" and not has_target_terms:
        mismatch_penalty += _term_penalty(
            service_text,
            _BROAD_EMPLOYMENT_TERMS,
            base=0.02,
            per=0.004,
            cap=0.035,
        )
    if not has_target_terms:
        mismatch_penalty += _term_penalty(
            service_text,
            _BROAD_SERVICE_TERMS,
            base=0.02,
            per=0.005,
            cap=0.035,
        )

    mismatch_penalty = min(0.065, mismatch_penalty)
    if mismatch_penalty > 0:
        adjustment -= mismatch_penalty
        reasons.append(f"theme:{theme}_mismatch_penalty")

    return adjustment, tuple(reasons)


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
        section for section in PRIMARY_SECTIONS if section_scores.get(section, 0.0) >= 0.55
    ]
    return min(MAX_PRIMARY_SECTION_BOOST, len(matched_primary_sections) * PRIMARY_SECTION_BOOST)


def _clamp_score(score: float) -> float:
    return min(1.0, max(0.0, score))


def _clamp_theme_adjustment(adjustment: float) -> float:
    return min(0.10, max(-0.11, adjustment))


def _service_level_metadata(
    candidates: Sequence[tuple[dict[str, str], float]],
) -> dict[str, str]:
    merged = dict(candidates[0][0])
    for metadata_field in _THEME_METADATA_FIELDS:
        values = [
            str(metadata.get(metadata_field, "")).strip()
            for metadata, _ in candidates
            if str(metadata.get(metadata_field, "")).strip()
        ]
        if values:
            merged[metadata_field] = " ".join(dict.fromkeys(values))

    for metadata_field in ("trgter_indvdl", "intrs_thema"):
        merged[metadata_field] = _json_str_list_as_metadata(
            metadata.get(metadata_field, "") for metadata, _ in candidates
        )
    return merged


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


def _service_theme_text(metadata: Mapping[str, object]) -> str:
    targets, themes, text = _metadata_terms(metadata)
    return " ".join((*targets, *themes, text))


def _service_text(metadata: Mapping[str, object]) -> str:
    return " ".join(str(metadata.get(field, "")) for field in ("serv_nm", "serv_dgst"))


def _json_str_list_as_metadata(values: Iterable[object]) -> str:
    unique: list[str] = []
    for value in values:
        for item in _json_str_list(value):
            if item not in unique:
                unique.append(item)
    return json.dumps(unique, ensure_ascii=False)


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


def _term_boost(
    text: str,
    terms: tuple[str, ...],
    *,
    base: float,
    per: float,
    cap: float,
) -> float:
    matches = _count_term_matches(text, terms)
    if matches == 0:
        return 0.0
    return min(cap, base + ((matches - 1) * per))


def _term_penalty(
    text: str,
    terms: tuple[str, ...],
    *,
    base: float,
    per: float,
    cap: float,
) -> float:
    matches = _count_term_matches(text, terms)
    if matches == 0:
        return 0.0
    return min(cap, base + ((matches - 1) * per))


def _count_term_matches(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term in text)


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

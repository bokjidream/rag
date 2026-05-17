from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from src.models.welfare import (
    EligibilityEvaluation,
    EligibilityEvidence,
    EligibilityStatus,
    SearchRequest,
)

_ELIGIBILITY_TEXT_FIELDS = ("tgtr_dtl_cn", "slct_crit_cn", "alw_serv_cn", "serv_dgst")
_STATUS_PRIORITY: dict[EligibilityStatus, int] = {
    "likely": 0,
    "needs_more_info": 1,
    "unlikely": 2,
}

_SENIOR_EXACT_PATTERNS = ("만 65세 이상", "독거노인", "노인2인가구")
_DISABILITY_PATTERNS = (
    "장애인활동지원 수급자",
    "장애인 가구",
    "장애의 정도가 심한 장애인",
    "등록 장애인",
    "중증 장애인",
    "장애인",
)
_CHILD_PATTERNS = ("부양자녀", "자녀장려금", "아동양육", "아동 양육", "아동양육비", "자녀 양육")
_EMPLOYMENT_PATTERNS = ("근로소득", "사업소득", "자활사업 참여", "자활참여", "구직활동")
_BASIC_OR_NEAR_POVERTY_PATTERNS = (
    "기초생활수급자",
    "국민기초생활보장",
    "조건부수급자",
    "일반수급자",
    "수급자",
    "차상위계층",
    "차상위 계층",
    "차상위자",
    "차상위",
)
_BROAD_LOW_INCOME_PATTERNS = ("저소득", "소득인정액", "기준 중위소득", "중위소득")


def evaluate_eligibility(
    request: SearchRequest,
    metadata: Mapping[str, object],
) -> EligibilityEvaluation:
    """Pattern-based eligibility guardrail.

    This is intentionally conservative: it only marks clear conflicts as unlikely and
    uses needs_more_info when the metadata suggests a possible but unconfirmed gap.
    """
    fields = _metadata_text_fields(metadata)
    missing_fields = [
        field for field in _ELIGIBILITY_TEXT_FIELDS if not fields.get(field, "").strip()
    ]

    hits: list[tuple[EligibilityStatus, str, EligibilityEvidence]] = []

    if request.age < 65:
        for pattern in _SENIOR_EXACT_PATTERNS:
            evidence = _find_evidence(fields, (pattern,))
            if evidence is not None:
                hits.append(
                    (
                        "unlikely",
                        f"{request.age}세 조건과 '{pattern}' 기준이 명백히 충돌합니다.",
                        evidence,
                    )
                )
                break

    if not request.disability:
        disability_evidence = _find_evidence(fields, _DISABILITY_PATTERNS)
        if disability_evidence is not None:
            if _is_disability_one_of_many(disability_evidence.text):
                hits.append(
                    (
                        "needs_more_info",
                        "장애인 조건이 여러 조건 중 하나로 보여 다른 자격 경로 확인이 필요합니다.",
                        disability_evidence,
                    )
                )
            else:
                hits.append(
                    (
                        "unlikely",
                        "장애인 조건이 핵심 필수 기준으로 보여 비장애인 조건과 충돌합니다.",
                        disability_evidence,
                    )
                )

    if request.has_children is False:
        child_evidence = _find_evidence(fields, _CHILD_PATTERNS)
        if child_evidence is not None:
            hits.append(
                (
                    "needs_more_info",
                    "자녀 또는 부양자녀 중심 기준이 있어 자녀 없음 조건 확인이 필요합니다.",
                    child_evidence,
                )
            )

    if request.employment_status == "비경제활동":
        employment_evidence = _find_evidence(fields, _EMPLOYMENT_PATTERNS)
        if employment_evidence is not None:
            hits.append(
                (
                    "needs_more_info",
                    "근로소득, 사업소득, 자활 또는 구직활동 기준 확인이 필요합니다.",
                    employment_evidence,
                )
            )

    if request.income_level == "저소득":
        income_evidence = _find_evidence(fields, _BASIC_OR_NEAR_POVERTY_PATTERNS)
        broad_low_income_evidence = _find_evidence(fields, _BROAD_LOW_INCOME_PATTERNS)
        if income_evidence is not None and broad_low_income_evidence is None:
            hits.append(
                (
                    "needs_more_info",
                    "기초생활수급자 또는 차상위계층 중심 기준으로 보여 저소득 해당 여부 확인이 필요합니다.",
                    income_evidence,
                )
            )

    if not hits:
        return EligibilityEvaluation(missing_fields=missing_fields)

    status = max((hit[0] for hit in hits), key=lambda item: _STATUS_PRIORITY[item])
    return EligibilityEvaluation(
        status=status,
        reasons=[reason for _, reason, _ in hits],
        missing_fields=missing_fields,
        evidence=_dedupe_evidence([evidence for _, _, evidence in hits]),
    )


def _metadata_text_fields(metadata: Mapping[str, object]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for field in _ELIGIBILITY_TEXT_FIELDS:
        value = metadata.get(field, "")
        fields[field] = value if isinstance(value, str) else str(value)
    return fields


def _find_evidence(
    fields: Mapping[str, str],
    patterns: Sequence[str],
) -> EligibilityEvidence | None:
    for field, text in fields.items():
        for pattern in patterns:
            if pattern in text:
                return EligibilityEvidence(field=field, text=_extract_context(text, pattern))
    return None


def _extract_context(text: str, pattern: str) -> str:
    for segment in re.split(r"(?<=[.!?。])\s+|\n+", text):
        cleaned = segment.strip()
        if pattern in cleaned:
            return cleaned
    return text.strip()


def _is_disability_one_of_many(text: str) -> bool:
    if any(marker in text for marker in ("아래 사항 중 하나", "다음 중 하나")):
        return True

    parts = [part.strip() for part in re.split(r",| 또는 | 혹은 ", text) if part.strip()]
    if len(parts) < 2:
        return False

    has_disability_part = any(any(pattern in part for pattern in _DISABILITY_PATTERNS) for part in parts)
    has_non_disability_part = any(
        not any(pattern in part for pattern in _DISABILITY_PATTERNS) for part in parts
    )
    return has_disability_part and has_non_disability_part


def _dedupe_evidence(evidence: Sequence[EligibilityEvidence]) -> list[EligibilityEvidence]:
    seen: set[tuple[str, str]] = set()
    deduped: list[EligibilityEvidence] = []
    for item in evidence:
        key = (item.field, item.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped

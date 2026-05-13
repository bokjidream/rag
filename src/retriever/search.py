from __future__ import annotations

import asyncio
import json
from typing import Any

from src.db.chroma import WELFARE_COLLECTION, get_collection
from src.embedding.protocol import EmbedderProtocol
from src.models.welfare import SearchRequest, SearchResponse, SearchResult, WelfareDetail


def _age_terms(age: int) -> list[str]:
    if age <= 5:
        return ["영유아", "아동", f"만 {age}세"]
    if age <= 18:
        return ["아동", "청소년", f"만 {age}세"]
    if age <= 34:
        return ["청년", f"만 {age}세"]
    if age < 65:
        return ["중장년", f"만 {age}세"]
    return ["노인", "어르신", "고령자", f"만 {age}세"]


def _income_terms(income_level: str) -> list[str]:
    terms_by_level = {
        "기초생활수급자": ["기초생활수급자", "국민기초생활보장", "수급자"],
        "차상위계층": ["차상위계층", "차상위"],
        "저소득": ["저소득", "저소득층", "소득인정액", "기준 중위소득 이하"],
        "일반": ["일반"],
    }
    return terms_by_level[income_level]


def _employment_terms(employment_status: str) -> list[str]:
    terms_by_status = {
        "취업": ["취업", "근로자", "재직"],
        "실업": ["실업", "구직", "취업 지원", "일자리", "직업훈련"],
        "비경제활동": ["비경제활동", "미취업"],
    }
    return terms_by_status[employment_status]


def _metadata_terms(metadata: dict[str, str]) -> tuple[list[str], list[str], str]:
    targets = json.loads(metadata["trgter_indvdl"])
    themes = json.loads(metadata["intrs_thema"])
    text = " ".join(
        [
            metadata["serv_nm"],
            metadata["serv_dgst"],
            metadata.get("tgtr_dtl_cn", ""),
            metadata.get("slct_crit_cn", ""),
            metadata.get("alw_serv_cn", ""),
        ]
    )
    return targets, themes, text


def _profile_boost(request: SearchRequest, metadata: dict[str, str]) -> float:
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


def _rank_score(request: SearchRequest, metadata: dict[str, str], distance: float) -> float:
    base_score = max(0.0, 1.0 - distance)
    return min(1.0, max(0.0, base_score + _profile_boost(request, metadata)))


def build_query_text(request: SearchRequest) -> str:
    """SearchRequest → 한국어 자연어 쿼리 문자열 변환."""
    parts: list[str] = [
        f"{request.age}세",
        *_age_terms(request.age),
        *_income_terms(request.income_level),
    ]
    if request.household_size is not None:
        parts.append(f"{request.household_size}인 가구")
    if request.marital_status is not None:
        parts.append(request.marital_status)
    if request.has_children is True:
        parts.extend(["미성년 자녀 있음", "자녀 양육", "아동", "보육"])
        if request.marital_status in {"이혼", "사별"}:
            parts.append("한부모 가족")
    if request.disability:
        parts.extend(["장애인", f"{request.disability_severity} 장애인"])
    if request.pregnant:
        parts.extend(["임산부", "임신", "출산", "산모"])
    if request.employment_status is not None:
        parts.extend(_employment_terms(request.employment_status))
    if request.region is not None:
        parts.append(request.region)
    return " ".join(parts) + " 거주자를 위한 복지 서비스"


async def search_welfare(
    request: SearchRequest,
    embedder: EmbedderProtocol,
) -> SearchResponse:
    """RAG 검색 메인 함수. SearchRequest → SearchResponse."""
    query_text = build_query_text(request)
    vec: list[float] = embedder.embed([query_text])[0]

    collection = await get_collection(WELFARE_COLLECTION)

    # chromadb query_embeddings 타입은 numpy array 기반 Union이라 list[list[float]]와
    # 불일치. 제로-인자 클로저로 감싸서 to_thread 호출 시 타입 검사를 우회한다.
    def _query() -> Any:
        return collection.query(
            query_embeddings=[vec],  # type: ignore[arg-type]
            n_results=min(request.top_k * 5, 100),
        )

    raw: Any = await asyncio.to_thread(_query)

    distances: list[float] = raw["distances"][0]
    metadatas: list[dict[str, str]] = raw["metadatas"][0]

    best_by_serv_id: dict[str, tuple[dict[str, str], float, float]] = {}
    for metadata, distance in zip(metadatas, distances):
        serv_id = metadata["serv_id"]
        score = _rank_score(request, metadata, distance)
        current = best_by_serv_id.get(serv_id)
        if current is None or score > current[2]:
            best_by_serv_id[serv_id] = (metadata, distance, score)

    ranked = sorted(best_by_serv_id.values(), key=lambda item: item[2], reverse=True)
    search_results = [
        SearchResult.from_metadata(metadata, distance, score=score)
        for metadata, distance, score in ranked[: request.top_k]
    ]

    return SearchResponse(results=search_results)


async def get_welfare_detail(serv_id: str) -> WelfareDetail | None:
    """ChromaDB에서 serv_id로 상세 정보 조회."""
    collection = await get_collection(WELFARE_COLLECTION)

    # chromadb Where 타입은 Literal 키 기반 중첩 구조라 dict[str, dict[str, str]]와
    # 불일치. 클로저로 감싸서 우회한다.
    def _get() -> Any:
        return collection.get(
            where={"serv_id": {"$eq": serv_id}},  # type: ignore[dict-item]
        )

    raw: Any = await asyncio.to_thread(_get)

    if not raw["ids"]:
        return None

    meta: dict[str, str] = raw["metadatas"][0]
    return WelfareDetail.from_metadata(meta)

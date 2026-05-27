from __future__ import annotations

import asyncio
from typing import Any

from src.db.chroma import WELFARE_COLLECTION, get_collection
from src.embedding.protocol import EmbedderProtocol
from src.models.welfare import SearchRequest, SearchResponse, SearchResult, WelfareDetail
from src.retriever import rerank as _rerank
from src.retriever.eligibility import evaluate_eligibility
from src.retriever.intent import QueryIntent, build_query_intent

DEFAULT_MAX_CANDIDATES = 100
ADAPTIVE_MAX_CANDIDATES = 500


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


def _profile_boost(request: SearchRequest, metadata: dict[str, str]) -> float:
    return _rerank._profile_boost(request, metadata)


def _rank_score(request: SearchRequest, metadata: dict[str, str], distance: float) -> float:
    return _rerank._rank_score(request, metadata, distance)


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
    *,
    collection_name: str = WELFARE_COLLECTION,
    adaptive_fetch: bool = False,
    max_candidates: int | None = None,
) -> SearchResponse:
    """RAG 검색 메인 함수. SearchRequest → SearchResponse."""
    query_text = build_query_text(request)
    vec: list[float] = embedder.embed([query_text])[0]

    collection = await get_collection(collection_name)
    candidate_limit = (
        ADAPTIVE_MAX_CANDIDATES
        if adaptive_fetch and max_candidates is None
        else DEFAULT_MAX_CANDIDATES
    )
    if max_candidates is not None:
        candidate_limit = max_candidates
    n_results = _initial_candidate_count(request.top_k, adaptive_fetch, candidate_limit)
    enable_section_rerank = collection_name != WELFARE_COLLECTION
    intent = build_query_intent(request)

    while True:
        raw = await _query_collection(collection, vec, n_results)
        search_results = _response_results_from_raw(
            request,
            raw,
            enable_section_rerank=enable_section_rerank,
            intent=intent,
        )

        if not adaptive_fetch or len(search_results) >= request.top_k or n_results >= candidate_limit:
            break
        next_n_results = min(candidate_limit, n_results * 2)
        if next_n_results == n_results:
            break
        n_results = next_n_results

    return SearchResponse(results=search_results)


def _initial_candidate_count(top_k: int, adaptive_fetch: bool, candidate_limit: int) -> int:
    if adaptive_fetch:
        return min(max(top_k * 20, 100), candidate_limit)
    return min(top_k * 10, candidate_limit)


async def _query_collection(
    collection: Any,
    query_vector: list[float],
    n_results: int,
) -> Any:
    # chromadb query_embeddings 타입은 numpy array 기반 Union이라 list[list[float]]와
    # 불일치. 제로-인자 클로저로 감싸서 to_thread 호출 시 타입 검사를 우회한다.
    def _query() -> Any:
        return collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
        )

    return await asyncio.to_thread(_query)


def _response_results_from_raw(
    request: SearchRequest,
    raw: Any,
    *,
    enable_section_rerank: bool = False,
    intent: QueryIntent | None = None,
) -> list[SearchResult]:
    distances: list[float] = raw["distances"][0]
    metadatas: list[dict[str, str]] = raw["metadatas"][0]
    query_intent = build_query_intent(request) if intent is None else intent
    use_section_rerank = enable_section_rerank and _has_section_aware_metadata(metadatas)

    chunks_by_serv_id: dict[str, list[tuple[dict[str, str], float]]] = {}
    for metadata, distance in zip(metadatas, distances):
        chunks_by_serv_id.setdefault(metadata["serv_id"], []).append((metadata, distance))

    ranked = sorted(
        (
            _rerank.rank_service_candidates(
                request,
                query_intent,
                candidates,
                enable_section_rerank=use_section_rerank,
            )
            for candidates in chunks_by_serv_id.values()
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    search_results: list[SearchResult] = []
    for service in ranked:
        result = SearchResult.from_metadata(service.metadata, service.distance, score=service.score)
        eligibility = evaluate_eligibility(request, service.metadata)
        if eligibility.status == "unlikely":
            continue
        search_results.append(
            result.model_copy(
                update={
                    "eligibility_status": eligibility.status,
                    "eligibility_reasons": eligibility.reasons,
                    "missing_fields": eligibility.missing_fields,
                    "evidence": eligibility.evidence,
                }
            )
        )
        if len(search_results) >= request.top_k:
            break
    return search_results


def _has_section_aware_metadata(metadatas: list[dict[str, str]]) -> bool:
    return any(metadata.get("chunk_section") for metadata in metadatas)


async def get_welfare_detail(
    serv_id: str,
    *,
    collection_name: str = WELFARE_COLLECTION,
) -> WelfareDetail | None:
    """ChromaDB에서 serv_id로 상세 정보 조회."""
    collection = await get_collection(collection_name)

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

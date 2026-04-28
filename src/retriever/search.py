from __future__ import annotations

import asyncio
import json
from typing import Any

from src.db.chroma import WELFARE_COLLECTION, get_collection
from src.embedding.protocol import EmbedderProtocol
from src.models.welfare import SearchRequest, SearchResponse, SearchResult, WelfareDetail


def build_query_text(request: SearchRequest) -> str:
    """SearchRequest → 한국어 자연어 쿼리 문자열 변환."""
    parts: list[str] = [f"{request.age}세", request.income_level]
    if request.household_size is not None:
        parts.append(f"{request.household_size}인 가구")
    if request.marital_status is not None:
        parts.append(request.marital_status)
    if request.has_children is True:
        parts.append("미성년 자녀 있음")
    if request.disability:
        if request.disability_severity:
            parts.append(f"장애인({request.disability_severity})")
        else:
            parts.append("장애인")
    if request.employment_status is not None:
        parts.append(request.employment_status)
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
            n_results=request.top_k,
        )

    raw: Any = await asyncio.to_thread(_query)

    distances: list[float] = raw["distances"][0]
    metadatas: list[dict[str, str]] = raw["metadatas"][0]

    search_results: list[SearchResult] = []
    for metadata, distance in zip(metadatas, distances):
        search_results.append(
            SearchResult(
                serv_id=metadata["serv_id"],
                serv_nm=metadata["serv_nm"],
                serv_dgst=metadata["serv_dgst"],
                department=metadata["jur_mnof_nm"],
                score=max(0.0, 1.0 - distance),
                trgter_indvdl=json.loads(metadata["trgter_indvdl"]),
                intrs_thema=json.loads(metadata["intrs_thema"]),
            )
        )

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
    return WelfareDetail(
        serv_id=meta["serv_id"],
        serv_nm=meta["serv_nm"],
        serv_dgst=meta["serv_dgst"],
        tgtr_dtl_cn=meta["tgtr_dtl_cn"],
        slct_crit_cn=meta["slct_crit_cn"],
        alw_serv_cn=meta["alw_serv_cn"],
        sprt_cyc_nm=meta["sprt_cyc_nm"],
        srv_pvsn_nm=meta["srv_pvsn_nm"],
        trgter_indvdl=json.loads(meta["trgter_indvdl"]),
        intrs_thema=json.loads(meta["intrs_thema"]),
        application_url=meta["serv_dtl_link"],
    )

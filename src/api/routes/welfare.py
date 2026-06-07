from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.config import ApiConfig
from src.api.deps import get_api_config, get_embedder
from src.embedding.protocol import EmbedderProtocol
from src.models.welfare import SearchRequest, SearchResponse, WelfareDetail
from src.retriever.search import get_welfare_detail, search_welfare

router = APIRouter(prefix="/welfare", tags=["welfare"])

ApiConfigDep = Annotated[ApiConfig, Depends(get_api_config)]
EmbedderDep = Annotated[EmbedderProtocol, Depends(get_embedder)]


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    embedder: EmbedderDep,
    api_config: ApiConfigDep,
) -> SearchResponse:
    """POST /welfare/search — 유저 조건으로 관련 서비스 top-k 검색."""
    return await search_welfare(
        request,
        embedder,
        collection_name=api_config.collection_name,
        adaptive_fetch=api_config.adaptive_fetch,
    )


@router.get("/{serv_id}", response_model=WelfareDetail)
async def get_detail(serv_id: str, api_config: ApiConfigDep) -> WelfareDetail:
    """GET /welfare/{serv_id} — 서비스 상세 정보 반환."""
    detail = await get_welfare_detail(
        serv_id,
        collection_name=api_config.collection_name,
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"서비스를 찾을 수 없습니다: {serv_id}",
        )
    return detail

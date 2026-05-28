from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.config import ApiConfig
from src.api.deps import get_api_config, get_embedder
from src.api.main import app
from src.models.welfare import SearchResponse, SearchResult, WelfareDetail

ROUTE_CONFIG = ApiConfig(
    collection_name="welfare_services_section_aware",
    adaptive_fetch=True,
)


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    mock_embedder = MagicMock()
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_api_config] = lambda: ROUTE_CONFIG
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_router_passes_same_resolved_collection_to_search_and_detail(
    client: AsyncClient,
) -> None:
    search_response = SearchResponse(
        results=[
            SearchResult(
                serv_id="WLF00000035",
                serv_nm="테스트 서비스",
                serv_dgst="서비스 개요",
                department="보건복지부",
                score=0.87,
                trgter_indvdl=["저소득"],
                intrs_thema=["생활지원"],
            )
        ]
    )
    detail_response = WelfareDetail(
        serv_id="WLF00000035",
        serv_nm="테스트 서비스",
        serv_dgst="서비스 개요",
        tgtr_dtl_cn="수급 대상 상세",
        slct_crit_cn="선정 기준",
        alw_serv_cn="서비스 내용",
        trgter_indvdl=["저소득"],
        intrs_thema=["생활지원"],
    )

    with (
        patch(
            "src.api.routes.welfare.search_welfare",
            new_callable=AsyncMock,
            return_value=search_response,
        ) as mock_search,
        patch(
            "src.api.routes.welfare.get_welfare_detail",
            new_callable=AsyncMock,
            return_value=detail_response,
        ) as mock_detail,
    ):
        search_http_response = await client.post(
            "/welfare/search",
            json={"age": 65, "income_level": "저소득"},
        )
        detail_http_response = await client.get("/welfare/WLF00000035")

    assert search_http_response.status_code == 200
    assert detail_http_response.status_code == 200
    assert mock_search.await_args.kwargs["collection_name"] == ROUTE_CONFIG.collection_name
    assert mock_search.await_args.kwargs["adaptive_fetch"] is ROUTE_CONFIG.adaptive_fetch
    assert mock_detail.await_args.kwargs["collection_name"] == ROUTE_CONFIG.collection_name

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_embedder
from src.api.main import app
from src.models.welfare import SearchResponse, SearchResult, WelfareDetail

MOCK_SEARCH_RESPONSE = SearchResponse(
    results=[
        SearchResult(
            serv_id="WLF00000035",
            serv_nm="테스트 서비스",
            serv_dgst="서비스 개요",
            department="국토교통부",
            score=0.87,
            trgter_indvdl=["저소득"],
            intrs_thema=["주거"],
        )
    ]
)

MOCK_WELFARE_DETAIL = WelfareDetail(
    serv_id="WLF00000035",
    serv_nm="테스트 서비스",
    serv_dgst="서비스 개요",
    tgtr_dtl_cn="수급 대상 상세",
    slct_crit_cn="선정 기준",
    alw_serv_cn="서비스 내용",
    sprt_cyc_nm="년",
    srv_pvsn_nm="현금지급",
    trgter_indvdl=["저소득"],
    intrs_thema=["주거"],
    application_url="https://bokjiro.go.kr/",
)


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    # ASGITransport은 lifespan을 트리거하지 않으므로 dependency_overrides로 embedder 주입
    mock_embedder = MagicMock()
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


class TestSearchEndpoint:
    async def test_valid_request_returns_200_and_search_response(
        self, client: AsyncClient
    ) -> None:
        with patch(
            "src.api.routes.welfare.search_welfare",
            new_callable=AsyncMock,
            return_value=MOCK_SEARCH_RESPONSE,
        ):
            response = await client.post(
                "/welfare/search",
                json={"age": 65, "income_level": "저소득"},
            )

        assert response.status_code == 200
        body = response.json()
        assert "results" in body
        assert len(body["results"]) == 1
        result = body["results"][0]
        assert result["serv_id"] == "WLF00000035"
        assert result["serv_nm"] == "테스트 서비스"
        assert result["score"] == pytest.approx(0.87)

    async def test_missing_required_field_returns_422(self, client: AsyncClient) -> None:
        response = await client.post(
            "/welfare/search",
            json={"age": 65},  # income_level 누락
        )
        assert response.status_code == 422

    async def test_disability_true_without_severity_returns_422(
        self, client: AsyncClient
    ) -> None:
        with patch(
            "src.api.routes.welfare.search_welfare",
            new_callable=AsyncMock,
        ):
            response = await client.post(
                "/welfare/search",
                json={
                    "age": 40,
                    "income_level": "저소득",
                    "disability": True,
                    "disability_severity": None,
                },
            )

        assert response.status_code == 422

    async def test_disability_true_with_severity_returns_200(
        self, client: AsyncClient
    ) -> None:
        with patch(
            "src.api.routes.welfare.search_welfare",
            new_callable=AsyncMock,
            return_value=MOCK_SEARCH_RESPONSE,
        ):
            response = await client.post(
                "/welfare/search",
                json={
                    "age": 40,
                    "income_level": "저소득",
                    "disability": True,
                    "disability_severity": "중증",
                },
            )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "payload",
        [
            {"age": -1, "income_level": "저소득"},
            {"age": 131, "income_level": "저소득"},
            {"age": 40, "income_level": "저소득", "top_k": 0},
            {"age": 40, "income_level": "저소득", "top_k": 51},
            {"age": 40, "income_level": "저소득", "household_size": 0},
            {
                "age": 40,
                "income_level": "저소득",
                "disability": False,
                "disability_severity": "중증",
            },
        ],
    )
    async def test_invalid_domain_request_returns_422(
        self, client: AsyncClient, payload: dict[str, object]
    ) -> None:
        response = await client.post("/welfare/search", json=payload)
        assert response.status_code == 422


class TestGetDetailEndpoint:
    async def test_existing_id_returns_200_and_welfare_detail(
        self, client: AsyncClient
    ) -> None:
        with patch(
            "src.api.routes.welfare.get_welfare_detail",
            new_callable=AsyncMock,
            return_value=MOCK_WELFARE_DETAIL,
        ):
            response = await client.get("/welfare/WLF00000035")

        assert response.status_code == 200
        body = response.json()
        assert body["serv_id"] == "WLF00000035"
        assert body["serv_nm"] == "테스트 서비스"
        assert body["tgtr_dtl_cn"] == "수급 대상 상세"
        assert body["required_documents"] == []
        assert body["application_method"] == ""
        assert body["application_forms"] == []
        assert "application_fields" not in body

    async def test_nonexistent_id_returns_404(self, client: AsyncClient) -> None:
        with patch(
            "src.api.routes.welfare.get_welfare_detail",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = await client.get("/welfare/NONEXISTENT")

        assert response.status_code == 404

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.welfare import SearchRequest, SearchResponse, WelfareDetail
from src.retriever.search import build_query_text, get_welfare_detail, search_welfare


class TestBuildQueryText:
    def test_required_fields_only(self) -> None:
        request = SearchRequest(age=65, income_level="저소득")
        result = build_query_text(request)
        assert "65세" in result
        assert "노인" in result
        assert "어르신" in result
        assert "저소득층" in result
        assert result.endswith("거주자를 위한 복지 서비스")

    def test_all_fields(self) -> None:
        request = SearchRequest(
            age=45,
            income_level="기초생활수급자",
            household_size=3,
            marital_status="기혼",
            has_children=True,
            disability=True,
            disability_severity="중증",
            employment_status="취업",
            region="경기도",
        )
        result = build_query_text(request)
        for term in [
            "45세",
            "중장년",
            "기초생활수급자",
            "국민기초생활보장",
            "3인 가구",
            "기혼",
            "자녀 양육",
            "중증 장애인",
            "근로자",
            "경기도",
        ]:
            assert term in result

    def test_has_children_false_excluded(self) -> None:
        request = SearchRequest(age=30, income_level="일반", has_children=False)
        result = build_query_text(request)
        assert "자녀" not in result

    def test_has_children_none_excluded(self) -> None:
        request = SearchRequest(age=30, income_level="일반", has_children=None)
        result = build_query_text(request)
        assert "자녀" not in result

    def test_disability_severity_included(self) -> None:
        request = SearchRequest(
            age=40, income_level="저소득", disability=True, disability_severity="경증"
        )
        result = build_query_text(request)
        assert "경증 장애인" in result

    def test_region_included(self) -> None:
        request = SearchRequest(age=65, income_level="저소득", region="서울")
        result = build_query_text(request)
        assert "서울" in result
        assert result.endswith("거주자를 위한 복지 서비스")

    def test_pregnant_adds_maternity_terms(self) -> None:
        request = SearchRequest(age=30, income_level="저소득", pregnant=True)
        result = build_query_text(request)
        assert "임산부" in result
        assert "임신" in result
        assert "출산" in result

    def test_not_pregnant_excludes_maternity_terms(self) -> None:
        request = SearchRequest(age=30, income_level="저소득")
        result = build_query_text(request)
        assert "임산부" not in result
        assert "임신" not in result


class TestSearchWelfare:
    @pytest.mark.asyncio
    async def test_returns_search_response(self) -> None:
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

        mock_query_result = {
            "ids": [["WLF001_chunk_0", "WLF002_chunk_0"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [
                [
                    {
                        "serv_id": "WLF001",
                        "serv_nm": "서비스1",
                        "serv_dgst": "개요1",
                        "jur_mnof_nm": "보건복지부",
                        "trgter_indvdl": json.dumps(["저소득"], ensure_ascii=False),
                        "intrs_thema": json.dumps(["생활지원"], ensure_ascii=False),
                    },
                    {
                        "serv_id": "WLF002",
                        "serv_nm": "서비스2",
                        "serv_dgst": "개요2",
                        "jur_mnof_nm": "국토교통부",
                        "trgter_indvdl": json.dumps(["노인"], ensure_ascii=False),
                        "intrs_thema": json.dumps(["주거"], ensure_ascii=False),
                    },
                ]
            ],
        }

        mock_collection = MagicMock()
        mock_collection.query.return_value = mock_query_result

        with patch("src.retriever.search.get_collection", new_callable=AsyncMock) as mock_get_col:
            mock_get_col.return_value = mock_collection

            request = SearchRequest(age=65, income_level="저소득", top_k=2)
            response = await search_welfare(request, mock_embedder)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 2

        first = response.results[0]
        assert first.serv_id == "WLF001"
        assert first.serv_nm == "서비스1"
        assert first.department == "보건복지부"
        assert first.score == pytest.approx(0.96)
        assert first.trgter_indvdl == ["저소득"]
        assert first.intrs_thema == ["생활지원"]

        second = response.results[1]
        assert second.score == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_score_clamped_to_zero(self) -> None:
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

        mock_query_result = {
            "ids": [["WLF001_chunk_0"]],
            "distances": [[1.5]],  # distance > 1이면 score < 0 → clamp to 0
            "metadatas": [
                [
                    {
                        "serv_id": "WLF001",
                        "serv_nm": "서비스1",
                        "serv_dgst": "개요1",
                        "jur_mnof_nm": "보건복지부",
                        "trgter_indvdl": json.dumps([]),
                        "intrs_thema": json.dumps([]),
                    }
                ]
            ],
        }

        mock_collection = MagicMock()
        mock_collection.query.return_value = mock_query_result

        with patch("src.retriever.search.get_collection", new_callable=AsyncMock) as mock_get_col:
            mock_get_col.return_value = mock_collection

            request = SearchRequest(age=30, income_level="일반")
            response = await search_welfare(request, mock_embedder)

        assert response.results[0].score == 0.0


class TestGetWelfareDetail:
    @pytest.mark.asyncio
    async def test_returns_welfare_detail_when_found(self) -> None:
        meta = {
            "serv_id": "WLF001",
            "serv_nm": "서비스명",
            "serv_dgst": "서비스 개요",
            "tgtr_dtl_cn": "수급 대상 상세",
            "slct_crit_cn": "선정 기준",
            "alw_serv_cn": "서비스 내용",
            "sprt_cyc_nm": "년",
            "srv_pvsn_nm": "현금지급",
            "trgter_indvdl": json.dumps(["저소득"], ensure_ascii=False),
            "intrs_thema": json.dumps(["주거"], ensure_ascii=False),
            "serv_dtl_link": "https://example.com",
        }
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["WLF001_chunk_0"],
            "metadatas": [meta],
        }

        with patch("src.retriever.search.get_collection", new_callable=AsyncMock) as mock_get_col:
            mock_get_col.return_value = mock_collection

            result = await get_welfare_detail("WLF001")

        assert isinstance(result, WelfareDetail)
        assert result.serv_id == "WLF001"
        assert result.serv_nm == "서비스명"
        assert result.tgtr_dtl_cn == "수급 대상 상세"
        assert result.trgter_indvdl == ["저소득"]
        assert result.intrs_thema == ["주거"]
        assert result.application_url == "https://example.com"
        assert result.required_documents == []
        assert result.application_method == ""
        assert result.application_forms == []

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self) -> None:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}

        with patch("src.retriever.search.get_collection", new_callable=AsyncMock) as mock_get_col:
            mock_get_col.return_value = mock_collection

            result = await get_welfare_detail("NONEXISTENT")

        assert result is None

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.welfare import WelfareRaw
from src.pipeline.index import index_welfare_items


class _MockEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _make_item(**kwargs: object) -> WelfareRaw:
    defaults: dict[str, object] = {
        "serv_id": "WLF001",
        "serv_nm": "테스트 서비스",
        "serv_dgst": "서비스 개요",
        "jur_mnof_nm": "보건복지부",
        "trgter_indvdl": ["저소득"],
        "intrs_thema": ["생활지원"],
        "sprt_cyc_nm": "월",
        "srv_pvsn_nm": "현금지급",
        "serv_dtl_link": "https://bokjiro.go.kr",
        "tgtr_dtl_cn": "수급 대상 상세",
        "slct_crit_cn": "선정 기준",
        "alw_serv_cn": "서비스 내용",
    }
    defaults.update(kwargs)
    return WelfareRaw(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_index_welfare_items_empty_list() -> None:
    mock_collection = MagicMock()
    with patch("src.pipeline.index.get_collection", new_callable=AsyncMock, return_value=mock_collection):
        result = await index_welfare_items([], _MockEmbedder())
    assert result == 0
    mock_collection.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_index_welfare_items_returns_positive_chunk_count() -> None:
    mock_collection = MagicMock()
    with patch("src.pipeline.index.get_collection", new_callable=AsyncMock, return_value=mock_collection):
        result = await index_welfare_items([_make_item()], _MockEmbedder())
    assert isinstance(result, int)
    assert result > 0


@pytest.mark.asyncio
async def test_index_welfare_items_calls_upsert_once() -> None:
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": []}
    with patch("src.pipeline.index.get_collection", new_callable=AsyncMock, return_value=mock_collection):
        await index_welfare_items([_make_item()], _MockEmbedder())
    mock_collection.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_index_welfare_items_chunk_id_format() -> None:
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": []}
    with patch("src.pipeline.index.get_collection", new_callable=AsyncMock, return_value=mock_collection):
        await index_welfare_items([_make_item(serv_id="WLF00000035")], _MockEmbedder())

    ids: list[str] = mock_collection.upsert.call_args.kwargs["ids"]
    for chunk_id in ids:
        assert chunk_id.startswith("WLF00000035_chunk_"), f"unexpected id: {chunk_id}"


@pytest.mark.asyncio
async def test_index_welfare_items_metadata_json_fields() -> None:
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": []}
    item = _make_item(
        trgter_indvdl=["저소득", "노인"],
        intrs_thema=["주거", "생활지원"],
        application_forms=[
            {
                "title": "보훈장학신청서(서식).hwp",
                "url": "https://bokjiro.go.kr/download/form.hwp",
                "file_type": "hwp",
            }
        ],
    )
    with patch("src.pipeline.index.get_collection", new_callable=AsyncMock, return_value=mock_collection):
        await index_welfare_items([item], _MockEmbedder())

    metadatas: list[dict[str, object]] = mock_collection.upsert.call_args.kwargs["metadatas"]
    for meta in metadatas:
        assert isinstance(meta["trgter_indvdl"], str)
        assert isinstance(meta["intrs_thema"], str)
        assert isinstance(meta["application_forms"], str)
        assert json.loads(str(meta["trgter_indvdl"])) == ["저소득", "노인"]
        assert json.loads(str(meta["intrs_thema"])) == ["주거", "생활지원"]
        assert json.loads(str(meta["application_forms"])) == [
            {
                "title": "보훈장학신청서(서식).hwp",
                "url": "https://bokjiro.go.kr/download/form.hwp",
                "file_type": "hwp",
            }
        ]


@pytest.mark.asyncio
async def test_index_welfare_items_metadata_includes_detail_fields() -> None:
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": []}
    item = _make_item(
        tgtr_dtl_cn="상세 대상 내용",
        slct_crit_cn="선정 기준 내용",
        alw_serv_cn="급여 내용",
        serv_dgst="서비스 요약",
    )
    with patch("src.pipeline.index.get_collection", new_callable=AsyncMock, return_value=mock_collection):
        await index_welfare_items([item], _MockEmbedder())

    metadatas: list[dict[str, object]] = mock_collection.upsert.call_args.kwargs["metadatas"]
    for meta in metadatas:
        assert meta["tgtr_dtl_cn"] == "상세 대상 내용"
        assert meta["slct_crit_cn"] == "선정 기준 내용"
        assert meta["alw_serv_cn"] == "급여 내용"
        assert meta["serv_dgst"] == "서비스 요약"


@pytest.mark.asyncio
async def test_index_welfare_items_multiple_items_one_upsert() -> None:
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": []}
    items = [_make_item(serv_id="WLF001"), _make_item(serv_id="WLF002")]
    with patch("src.pipeline.index.get_collection", new_callable=AsyncMock, return_value=mock_collection):
        result = await index_welfare_items(items, _MockEmbedder())

    assert result > 0
    # 모든 청크를 한 번의 upsert로 처리 (배치 효율화)
    mock_collection.upsert.assert_called_once()
    ids: list[str] = mock_collection.upsert.call_args.kwargs["ids"]
    # WLF001, WLF002 양쪽의 청크 ID가 모두 있어야 함
    assert any("WLF001" in i for i in ids)
    assert any("WLF002" in i for i in ids)


@pytest.mark.asyncio
async def test_index_welfare_items_deletes_stale_chunk_ids() -> None:
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["WLF001_chunk_0", "WLF001_chunk_1", "OLD_chunk_0"],
    }
    with patch("src.pipeline.index.get_collection", new_callable=AsyncMock, return_value=mock_collection):
        await index_welfare_items([_make_item(serv_id="WLF001")], _MockEmbedder())

    ids: list[str] = mock_collection.upsert.call_args.kwargs["ids"]
    expected_stale = ["OLD_chunk_0"]
    if "WLF001_chunk_1" not in ids:
        expected_stale.append("WLF001_chunk_1")
    mock_collection.delete.assert_called_once_with(ids=expected_stale)

"""src/crawler/client.py 단위 테스트 — 실제 네트워크 요청 없음."""
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.crawler.client import fetch_welfare_detail, fetch_welfare_list
from tests.crawler.conftest import SAMPLE_DETAIL_001, SAMPLE_LIST_PAGE1


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELFARE_API_KEY", "test_key")


def _make_mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


# ── fetch_welfare_list ──────────────────────────────────────────────


async def test_fetch_welfare_list_returns_list_payload() -> None:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = _make_mock_response(SAMPLE_LIST_PAGE1)

    result = await fetch_welfare_list(mock_client, page=1)

    assert result == SAMPLE_LIST_PAGE1


async def test_fetch_welfare_list_sends_correct_params() -> None:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = _make_mock_response(SAMPLE_LIST_PAGE1)

    await fetch_welfare_list(mock_client, page=2, num_rows=50)

    _, kwargs = mock_client.get.call_args
    params = kwargs["params"]
    assert params["callTp"] == "R"
    assert params["pageNo"] == "2"
    assert params["numOfRows"] == "50"
    assert params["serviceKey"] == "test_key"


async def test_fetch_welfare_list_raises_on_http_error() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = mock_resp

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_welfare_list(mock_client)


# ── fetch_welfare_detail ────────────────────────────────────────────


async def test_fetch_welfare_detail_returns_detail_payload() -> None:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = _make_mock_response(SAMPLE_DETAIL_001)

    result = await fetch_welfare_detail(mock_client, "WLF00000001")

    assert result == SAMPLE_DETAIL_001


async def test_fetch_welfare_detail_sends_serv_id() -> None:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = _make_mock_response(SAMPLE_DETAIL_001)

    await fetch_welfare_detail(mock_client, "WLF00000001")

    _, kwargs = mock_client.get.call_args
    assert kwargs["params"]["servId"] == "WLF00000001"
    assert kwargs["params"]["callTp"] == "R"


async def test_fetch_welfare_detail_raises_on_http_error() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=MagicMock()
    )
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = mock_resp

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_welfare_detail(mock_client, "INVALID")

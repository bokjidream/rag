"""src/crawler/collect.py 단위 테스트 — 실제 네트워크 요청 없음."""
import json
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.crawler.collect import collect_all
from tests.crawler.conftest import SAMPLE_DETAIL_001, SAMPLE_DETAIL_002, SAMPLE_LIST_PAGE1

DETAIL_MAP = {
    "WLF00000001": SAMPLE_DETAIL_001,
    "WLF00000002": SAMPLE_DETAIL_002,
}


def _make_mock_client(detail_map: dict, fail_ids: Optional[set] = None) -> AsyncMock:
    """목록·상세 API를 모킹하는 httpx.AsyncClient 반환."""
    fail_ids = fail_ids or set()

    async def mock_get(url: str, **kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "NationalWelfarelistV001" in url:
            resp.json.return_value = SAMPLE_LIST_PAGE1
        else:
            serv_id = kwargs["params"]["servId"]  # type: ignore[index]
            if serv_id in fail_ids:
                resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "500", request=MagicMock(), response=MagicMock()
                )
            else:
                resp.json.return_value = detail_map[serv_id]
        return resp

    mock_client = AsyncMock()
    mock_client.get.side_effect = mock_get
    return mock_client


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WELFARE_API_KEY", "test_key")


# ── collect_all ─────────────────────────────────────────────────────


async def test_collect_all_returns_all_services(tmp_path: Path) -> None:
    mock_client = _make_mock_client(DETAIL_MAP)

    with patch("src.crawler.collect.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        services = await collect_all(output_path=tmp_path / "out.json")

    assert len(services) == 2
    serv_ids = {s["servId"] for s in services}
    assert serv_ids == {"WLF00000001", "WLF00000002"}


async def test_collect_all_skips_failed_serv_id(tmp_path: Path) -> None:
    """API 실패한 servId는 skip하고 나머지 계속 수집."""
    mock_client = _make_mock_client(DETAIL_MAP, fail_ids={"WLF00000001"})

    with patch("src.crawler.collect.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        services = await collect_all(output_path=tmp_path / "out.json")

    assert len(services) == 1
    assert services[0]["servId"] == "WLF00000002"


async def test_collect_all_saves_json(tmp_path: Path) -> None:
    """수집 결과가 JSON 파일로 저장된다."""
    output_path = tmp_path / "welfare_services.json"
    mock_client = _make_mock_client(DETAIL_MAP)

    with patch("src.crawler.collect.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        await collect_all(output_path=output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved) == 2


async def test_collect_all_creates_parent_dir(tmp_path: Path) -> None:
    """output_path 상위 디렉토리가 없어도 자동 생성."""
    output_path = tmp_path / "nested" / "dir" / "out.json"
    mock_client = _make_mock_client(DETAIL_MAP)

    with patch("src.crawler.collect.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        await collect_all(output_path=output_path)

    assert output_path.exists()

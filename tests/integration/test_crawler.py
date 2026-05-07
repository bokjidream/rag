from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.models.welfare import WelfareRaw

LIST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<servList>
  <servInfo>
    <servId>WLF00000001</servId>
    <servNm>노인 기초연금</servNm>
    <servDgst>65세 이상 저소득 어르신 연금 지원</servDgst>
    <jurMnofNm>보건복지부</jurMnofNm>
    <trgterIndvdlArray>저소득,노인</trgterIndvdlArray>
    <intrsThemaArray>소득지원,노인복지</intrsThemaArray>
    <sprtCycNm>월</sprtCycNm>
    <srvPvsnNm>현금지급</srvPvsnNm>
    <servDtlLink>https://bokjiro.go.kr/ssis-tbu/twataa/wlfareInfo/moveTWAT52011M.do?wlfareInfoId=WLF00000001</servDtlLink>
  </servInfo>
  <servInfo>
    <servId>WLF00000002</servId>
    <servNm>장애인 활동지원</servNm>
    <servDgst>장애인 활동보조 지원</servDgst>
    <jurMnofNm>보건복지부</jurMnofNm>
    <trgterIndvdlArray>장애인</trgterIndvdlArray>
    <intrsThemaArray>장애인복지</intrsThemaArray>
    <sprtCycNm>월</sprtCycNm>
    <srvPvsnNm>서비스</srvPvsnNm>
    <servDtlLink>https://bokjiro.go.kr/ssis-tbu/twataa/wlfareInfo/moveTWAT52011M.do?wlfareInfoId=WLF00000002</servDtlLink>
  </servInfo>
</servList>
""".encode()

DETAIL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<servDtl>
  <servInfo>
    <servId>WLF00000001</servId>
    <tgtrDtlCn>65세 이상 어르신 중 소득 하위 70%</tgtrDtlCn>
    <slctCritCn>소득인정액 기준 이하</slctCritCn>
    <alwServCn>매월 최대 32만원 현금 지급</alwServCn>
  </servInfo>
  <basfrmList>
    <servSeDetailLink>https://bokjiro.go.kr/download/form.hwp</servSeDetailLink>
    <servSeDetailNm>기초연금 신청서.hwp</servSeDetailNm>
  </basfrmList>
  <basfrmList>
    <servSeDetailLink>https://bokjiro.go.kr/download/guide.pdf</servSeDetailLink>
    <servSeDetailNm>2026년 기초연금 사업안내.pdf</servSeDetailNm>
  </basfrmList>
</servDtl>
""".encode()

EMPTY_ARRAY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<servList>
  <servInfo>
    <servId>WLF00000003</servId>
    <servNm>테스트 서비스</servNm>
    <servDgst>테스트 개요</servDgst>
    <jurMnofNm>테스트부</jurMnofNm>
    <trgterIndvdlArray></trgterIndvdlArray>
    <intrsThemaArray></intrsThemaArray>
    <sprtCycNm>년</sprtCycNm>
    <srvPvsnNm>현물</srvPvsnNm>
    <servDtlLink>https://example.com</servDtlLink>
  </servInfo>
</servList>
""".encode()

# XML where items are <servList> elements (no <servInfo>) — exercises fallback branch
SERV_LIST_FALLBACK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<response>
  <servList>
    <servId>WLF00000099</servId>
    <servNm>낙상 예방 서비스</servNm>
    <servDgst>낙상 위험군 예방 지원</servDgst>
    <jurMnofNm>보건복지부</jurMnofNm>
    <trgterIndvdlArray>노인</trgterIndvdlArray>
    <intrsThemaArray>노인복지</intrsThemaArray>
    <sprtCycNm>년</sprtCycNm>
    <srvPvsnNm>서비스</srvPvsnNm>
    <servDtlLink>https://example.com</servDtlLink>
  </servList>
</response>
""".encode()

# XML missing some fields — exercises _text(None) → "" branch
MISSING_FIELDS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<servList>
  <servInfo>
    <servId>WLF00000004</servId>
    <servNm>필드 누락 서비스</servNm>
  </servInfo>
</servList>
""".encode()

MISSING_DETAIL_FIELDS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<servDtl>
  <servInfo>
    <servId>WLF00000004</servId>
  </servInfo>
</servDtl>
"""


def _make_response(content: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


# ──────────────────────────────────────────────
# fetch_welfare_list
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_welfare_list_parses_fields() -> None:
    """목록 API XML 응답에서 WelfareRaw 필드를 정상 파싱한다."""
    from src.crawler.welfare_list import fetch_welfare_list

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_make_response(LIST_XML))

    results = await fetch_welfare_list(api_key="test_key", client=mock_client)

    assert len(results) == 2
    first = results[0]
    assert first["serv_id"] == "WLF00000001"
    assert first["serv_nm"] == "노인 기초연금"
    assert first["serv_dgst"] == "65세 이상 저소득 어르신 연금 지원"
    assert first["jur_mnof_nm"] == "보건복지부"
    assert first["sprt_cyc_nm"] == "월"
    assert first["srv_pvsn_nm"] == "현금지급"
    assert "bokjiro.go.kr" in first["serv_dtl_link"]


@pytest.mark.asyncio
async def test_fetch_welfare_list_parses_array_fields() -> None:
    """trgterIndvdlArray, intrsThemaArray 콤마 구분 문자열을 list[str]로 변환한다."""
    from src.crawler.welfare_list import fetch_welfare_list

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_make_response(LIST_XML))

    results = await fetch_welfare_list(api_key="test_key", client=mock_client)

    assert results[0]["trgter_indvdl"] == ["저소득", "노인"]
    assert results[0]["intrs_thema"] == ["소득지원", "노인복지"]
    assert results[1]["trgter_indvdl"] == ["장애인"]


@pytest.mark.asyncio
async def test_fetch_welfare_list_empty_arrays() -> None:
    """빈 trgterIndvdlArray, intrsThemaArray는 빈 리스트로 처리한다."""
    from src.crawler.welfare_list import fetch_welfare_list

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_make_response(EMPTY_ARRAY_XML))

    results = await fetch_welfare_list(api_key="test_key", client=mock_client)

    assert results[0]["trgter_indvdl"] == []
    assert results[0]["intrs_thema"] == []


# ──────────────────────────────────────────────
# fetch_welfare_detail
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_welfare_detail_parses_fields() -> None:
    """상세 API XML 응답에서 상세 필드를 정상 파싱한다."""
    from src.crawler.welfare_detail import fetch_welfare_detail

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_make_response(DETAIL_XML))

    result = await fetch_welfare_detail(serv_id="WLF00000001", api_key="test_key", client=mock_client)

    assert result["tgtr_dtl_cn"] == "65세 이상 어르신 중 소득 하위 70%"
    assert result["slct_crit_cn"] == "소득인정액 기준 이하"
    assert result["alw_serv_cn"] == "매월 최대 32만원 현금 지급"
    assert result["application_forms"] == [
        {
            "title": "기초연금 신청서.hwp",
            "url": "https://bokjiro.go.kr/download/form.hwp",
            "file_type": "hwp",
        },
        {
            "title": "2026년 기초연금 사업안내.pdf",
            "url": "https://bokjiro.go.kr/download/guide.pdf",
            "file_type": "pdf",
        },
    ]


# ──────────────────────────────────────────────
# collect_all
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collect_all_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """PUBLIC_DATA_API_KEY 환경변수가 없으면 ValueError를 발생시킨다."""
    monkeypatch.delenv("PUBLIC_DATA_API_KEY", raising=False)

    from src.crawler.collect import collect_all

    with pytest.raises(ValueError, match="PUBLIC_DATA_API_KEY"):
        await collect_all()


@pytest.mark.asyncio
async def test_collect_all_returns_empty_when_detail_crawler_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """상세 크롤러 결과가 비어 있으면 빈 리스트를 반환한다."""
    monkeypatch.setenv("PUBLIC_DATA_API_KEY", "test_key")
    monkeypatch.setenv("APPLICATION_FORMS_API_LIMIT", "0")
    monkeypatch.setenv("APPLICATION_FORMS_STATE_PATH", str(tmp_path / "forms_state.json"))

    list_item: dict[str, Any] = {
        "serv_id": "WLF00000001",
        "serv_nm": "노인 기초연금",
        "serv_dgst": "65세 이상 저소득 어르신 연금 지원",
        "jur_mnof_nm": "보건복지부",
        "trgter_indvdl": ["저소득", "노인"],
        "intrs_thema": ["소득지원"],
        "sprt_cyc_nm": "월",
        "srv_pvsn_nm": "현금지급",
        "serv_dtl_link": "https://example.com",
    }

    async def mock_fetch_list(
        api_key: str,
        page: int = 1,
        per_page: int = 100,
        client: Any = None,
    ) -> list[dict[str, Any]]:
        if page == 1:
            return [list_item]
        return []

    mock_crawl_details = AsyncMock(return_value=[])

    from src.crawler.collect import collect_all

    with (
        patch("src.crawler.collect.fetch_welfare_list", side_effect=mock_fetch_list),
        patch("src.crawler.collect._crawl_details", mock_crawl_details),
    ):
        results = await collect_all(max_pages=1)

    assert results == []
    mock_crawl_details.assert_awaited_once_with([list_item])


@pytest.mark.asyncio
async def test_collect_all_returns_welfare_raw_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """정상 응답일 때 WelfareRaw 객체 리스트를 반환한다."""
    monkeypatch.setenv("PUBLIC_DATA_API_KEY", "test_key")
    monkeypatch.setenv("APPLICATION_FORMS_API_LIMIT", "0")
    monkeypatch.setenv("APPLICATION_FORMS_STATE_PATH", str(tmp_path / "forms_state.json"))

    list_item: dict[str, Any] = {
        "serv_id": "WLF00000001",
        "serv_nm": "노인 기초연금",
        "serv_dgst": "65세 이상 저소득 어르신 연금 지원",
        "jur_mnof_nm": "보건복지부",
        "trgter_indvdl": ["저소득", "노인"],
        "intrs_thema": ["소득지원"],
        "sprt_cyc_nm": "월",
        "srv_pvsn_nm": "현금지급",
        "serv_dtl_link": "https://example.com",
    }
    detail_item: dict[str, Any] = {
        "tgtr_dtl_cn": "65세 이상",
        "slct_crit_cn": "소득 기준 이하",
        "alw_serv_cn": "매월 32만원",
    }
    crawled_items = [WelfareRaw(**{**list_item, **detail_item})]

    async def mock_fetch_list(
        api_key: str,
        page: int = 1,
        per_page: int = 100,
        client: Any = None,
    ) -> list[dict[str, Any]]:
        if page == 1:
            return [list_item]
        return []

    mock_crawl_details = AsyncMock(return_value=crawled_items)

    from src.crawler.collect import collect_all

    with (
        patch("src.crawler.collect.fetch_welfare_list", side_effect=mock_fetch_list),
        patch("src.crawler.collect._crawl_details", mock_crawl_details),
    ):
        results = await collect_all(max_pages=1)

    assert len(results) == 1
    item = results[0]
    assert isinstance(item, WelfareRaw)
    assert item.serv_id == "WLF00000001"
    assert item.tgtr_dtl_cn == "65세 이상"
    assert item.slct_crit_cn == "소득 기준 이하"
    assert item.alw_serv_cn == "매월 32만원"
    mock_crawl_details.assert_awaited_once_with([list_item])


# ──────────────────────────────────────────────
# build_client
# ──────────────────────────────────────────────


def test_build_client_returns_async_client() -> None:
    from src.crawler.client import build_client

    client = build_client()
    assert isinstance(client, httpx.AsyncClient)


# ──────────────────────────────────────────────
# _text(None) — missing field fallback
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_welfare_list_missing_fields_returns_empty_strings() -> None:
    """XML에 일부 필드가 없을 때 _text(None) → "" 경로를 커버한다."""
    from src.crawler.welfare_list import fetch_welfare_list

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_make_response(MISSING_FIELDS_XML))

    results = await fetch_welfare_list(api_key="test_key", client=mock_client)

    assert results[0]["serv_dgst"] == ""
    assert results[0]["jur_mnof_nm"] == ""


@pytest.mark.asyncio
async def test_fetch_welfare_detail_missing_fields_returns_empty_strings() -> None:
    """XML에 상세 필드가 없을 때 _text(None) → "" 경로를 커버한다."""
    from src.crawler.welfare_detail import fetch_welfare_detail

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_make_response(MISSING_DETAIL_FIELDS_XML))

    result = await fetch_welfare_detail(serv_id="WLF00000004", api_key="test_key", client=mock_client)

    assert result["tgtr_dtl_cn"] == ""
    assert result["slct_crit_cn"] == ""
    assert result["alw_serv_cn"] == ""
    assert result["application_forms"] == []


@pytest.mark.asyncio
async def test_enrich_application_forms_uses_cached_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """API 호출 제한이 0이어도 기존 fetch 상태 파일의 신청서 자료는 반영한다."""
    state_path = tmp_path / "forms_state.json"
    cached_forms = [
        {
            "title": "보훈장학신청서(서식).hwp",
            "url": "https://bokjiro.go.kr/download/form.hwp",
            "file_type": "hwp",
        }
    ]
    state_path.write_text(
        json.dumps(
            {
                "WLF00000054": {
                    "status": "success",
                    "fetched_at": "2026-05-07T00:00:00+00:00",
                    "application_forms": cached_forms,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("APPLICATION_FORMS_API_LIMIT", "0")
    monkeypatch.setenv("APPLICATION_FORMS_STATE_PATH", str(state_path))

    from src.crawler.collect import _enrich_application_forms

    results = await _enrich_application_forms(
        [{"serv_id": "WLF00000054", "serv_nm": "보훈장학금"}],
        api_key="test_key",
    )

    assert results[0]["application_forms"] == cached_forms


@pytest.mark.asyncio
async def test_enrich_application_forms_fetches_up_to_limit_and_records_empty_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """상세 API 호출 수를 제한하고, basfrmList가 없는 서비스도 성공 상태로 기록한다."""
    state_path = tmp_path / "forms_state.json"
    monkeypatch.setenv("APPLICATION_FORMS_API_LIMIT", "2")
    monkeypatch.setenv("APPLICATION_FORMS_STATE_PATH", str(state_path))
    monkeypatch.setattr("src.crawler.collect._APPLICATION_FORMS_FETCH_DELAY_SECONDS", 0)

    async def mock_fetch_detail(
        serv_id: str,
        api_key: str,
        client: Any = None,
    ) -> dict[str, Any]:
        if serv_id == "WLF00001093":
            return {
                "application_forms": [
                    {
                        "title": "응급안전안심서비스 신청서.hwp",
                        "url": "https://bokjiro.go.kr/download/emergency.hwp",
                        "file_type": "hwp",
                    }
                ]
            }
        return {"application_forms": []}

    mock_client = AsyncMock()

    @asynccontextmanager  # type: ignore[misc]
    async def _mock_build_client(**_: object):  # type: ignore[misc]
        yield mock_client

    from src.crawler.collect import _enrich_application_forms

    with (
        patch("src.crawler.collect.build_client", _mock_build_client),
        patch("src.crawler.collect.fetch_welfare_detail", side_effect=mock_fetch_detail) as mock_fetch,
    ):
        results = await _enrich_application_forms(
            [
                {"serv_id": "WLF00001093", "serv_nm": "응급안전안심서비스"},
                {"serv_id": "WLF00000035", "serv_nm": "개발제한구역 생활비용보조"},
                {"serv_id": "WLF00000054", "serv_nm": "보훈장학금"},
            ],
            api_key="test_key",
        )

    assert mock_fetch.await_count == 2
    assert results[0]["application_forms"] == [
        {
            "title": "응급안전안심서비스 신청서.hwp",
            "url": "https://bokjiro.go.kr/download/emergency.hwp",
            "file_type": "hwp",
        }
    ]
    assert results[1]["application_forms"] == []
    assert "application_forms" not in results[2]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["WLF00001093"]["status"] == "success"
    assert state["WLF00000035"]["application_forms"] == []
    assert "WLF00000054" not in state


# ──────────────────────────────────────────────
# servList fallback
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_welfare_list_servlist_fallback() -> None:
    """servInfo 요소가 없을 때 servList 요소로 폴백한다."""
    from src.crawler.welfare_list import fetch_welfare_list

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=_make_response(SERV_LIST_FALLBACK_XML))

    results = await fetch_welfare_list(api_key="test_key", client=mock_client)

    assert len(results) == 1
    assert results[0]["serv_id"] == "WLF00000099"


# ──────────────────────────────────────────────
# no-client path
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_welfare_list_no_client_uses_build_client() -> None:
    """client 인자 없이 호출하면 내부에서 build_client()를 사용한다."""
    from src.crawler.welfare_list import fetch_welfare_list

    mock_inner = AsyncMock()
    mock_inner.get = AsyncMock(return_value=_make_response(LIST_XML))

    @asynccontextmanager  # type: ignore[misc]
    async def _mock_build_client(**_: object):  # type: ignore[misc]
        yield mock_inner

    with patch("src.crawler.client.build_client", _mock_build_client):
        results = await fetch_welfare_list(api_key="test_key")

    assert len(results) == 2


@pytest.mark.asyncio
async def test_fetch_welfare_detail_no_client_uses_build_client() -> None:
    """client 인자 없이 호출하면 내부에서 build_client()를 사용한다."""
    from src.crawler.welfare_detail import fetch_welfare_detail

    mock_inner = AsyncMock()
    mock_inner.get = AsyncMock(return_value=_make_response(DETAIL_XML))

    @asynccontextmanager  # type: ignore[misc]
    async def _mock_build_client(**_: object):  # type: ignore[misc]
        yield mock_inner

    with patch("src.crawler.client.build_client", _mock_build_client):
        result = await fetch_welfare_detail(serv_id="WLF00000001", api_key="test_key")

    assert result["tgtr_dtl_cn"] == "65세 이상 어르신 중 소득 하위 70%"


# ──────────────────────────────────────────────
# Integration tests (require real API key)
# ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_welfare_list_real_api() -> None:
    """실제 공공데이터포털 API를 호출하여 목록을 가져온다 (CI 제외)."""
    api_key = os.environ.get("PUBLIC_DATA_API_KEY")
    if not api_key:
        pytest.skip("PUBLIC_DATA_API_KEY not set")

    from src.crawler.client import build_client
    from src.crawler.welfare_list import fetch_welfare_list

    async with build_client() as client:
        results = await fetch_welfare_list(api_key=api_key, page=1, per_page=5, client=client)

    assert len(results) > 0
    assert "serv_id" in results[0]

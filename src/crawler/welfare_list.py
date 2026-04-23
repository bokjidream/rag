from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import httpx

LIST_URL = (
    "https://apis.data.go.kr/B554287/NationalWelfareInformationService"
    "/NationalWelfarelistInquiry"
)


def _parse_array(text: str | None) -> list[str]:
    if not text or not text.strip():
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _parse_list_xml(content: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(content)
    results: list[dict[str, Any]] = []
    for info in root.findall("servInfo"):
        results.append(
            {
                "serv_id": _text(info.find("servId")),
                "serv_nm": _text(info.find("servNm")),
                "serv_dgst": _text(info.find("servDgst")),
                "jur_mnof_nm": _text(info.find("jurMnofNm")),
                "trgter_indvdl": _parse_array(_text(info.find("trgterIndvdlArray"))),
                "intrs_thema": _parse_array(_text(info.find("intrsThemaArray"))),
                "sprt_cyc_nm": _text(info.find("sprtCycNm")),
                "srv_pvsn_nm": _text(info.find("srvPvsnNm")),
                "serv_dtl_link": _text(info.find("servDtlLink")),
            }
        )
    return results


async def fetch_welfare_list(
    api_key: str,
    page: int = 1,
    per_page: int = 100,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    params = {
        "serviceKey": api_key,
        "callTp": "L",
        "pageIndex": str(page),
        "numOfRows": str(per_page),
    }

    async def _fetch(c: httpx.AsyncClient) -> list[dict[str, Any]]:
        response = await c.get(LIST_URL, params=params)
        response.raise_for_status()
        return _parse_list_xml(response.content)

    if client is not None:
        return await _fetch(client)

    from src.crawler.client import build_client

    async with build_client() as c:
        return await _fetch(c)

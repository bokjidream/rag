from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import httpx

DETAIL_URL = (
    "https://apis.data.go.kr/B554287/NationalWelfareInformationsV001"
    "/NationalWelfaredetailedV001"
)


def _text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _parse_detail_xml(content: bytes) -> dict[str, Any]:
    root = ET.fromstring(content)
    return {
        "tgtr_dtl_cn": _text(root.find(".//tgtrDtlCn")),
        "slct_crit_cn": _text(root.find(".//slctCritCn")),
        "alw_serv_cn": _text(root.find(".//alwServCn")),
    }


async def fetch_welfare_detail(
    serv_id: str,
    api_key: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    params = {
        "serviceKey": api_key,
        "callTp": "D",
        "servId": serv_id,
    }

    async def _fetch(c: httpx.AsyncClient) -> dict[str, Any]:
        response = await c.get(DETAIL_URL, params=params)
        response.raise_for_status()
        return _parse_detail_xml(response.content)

    if client is not None:
        return await _fetch(client)

    from src.crawler.client import build_client

    async with build_client() as c:
        return await _fetch(c)

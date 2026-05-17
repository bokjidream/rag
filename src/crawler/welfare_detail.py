from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Literal

import httpx

DETAIL_URL = (
    "https://apis.data.go.kr/B554287/NationalWelfareInformationsV001/NationalWelfaredetailedV001"
)


def _text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _detect_file_type(title: str, url: str) -> Literal["pdf", "hwp", "hwpx", "etc"]:
    haystack = f"{title} {url}".lower()
    for file_type in ("hwpx", "hwp", "pdf"):
        if f".{file_type}" in haystack:
            return file_type
    return "etc"


def _parse_application_forms(root: ET.Element) -> list[dict[str, str]]:
    forms: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for form in root.findall(".//basfrmList"):
        title = _text(form.find("servSeDetailNm"))
        url = _text(form.find("servSeDetailLink"))
        if not title and not url:
            continue

        key = (title, url)
        if key in seen:
            continue
        seen.add(key)

        forms.append(
            {
                "title": title,
                "url": url,
                "file_type": _detect_file_type(title, url),
            }
        )

    return forms


def _parse_detail_xml(content: bytes) -> dict[str, Any]:
    root = ET.fromstring(content)
    return {
        "tgtr_dtl_cn": _text(root.find(".//tgtrDtlCn")),
        "slct_crit_cn": _text(root.find(".//slctCritCn")),
        "alw_serv_cn": _text(root.find(".//alwServCn")),
        "application_forms": _parse_application_forms(root),
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

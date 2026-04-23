from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET

import httpx

from src.crawler.welfare_detail import fetch_welfare_detail
from src.crawler.welfare_list import fetch_welfare_list
from src.models.welfare import WelfareRaw

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def collect_all(max_pages: int = 10, per_page: int = 100) -> list[WelfareRaw]:
    api_key = os.environ.get("PUBLIC_DATA_API_KEY")
    if not api_key:
        raise ValueError("PUBLIC_DATA_API_KEY 환경변수가 설정되지 않았습니다.")

    results: list[WelfareRaw] = []

    for page in range(1, max_pages + 1):
        try:
            items = await fetch_welfare_list(api_key=api_key, page=page, per_page=per_page)
        except (httpx.HTTPError, httpx.TimeoutException, ET.ParseError, KeyError, ValueError) as e:
            logger.warning("목록 페이지 %d 스킵: %s", page, str(e))
            break

        for item in items:
            serv_id = item.get("serv_id", "")
            try:
                detail = await fetch_welfare_detail(serv_id=serv_id, api_key=api_key)
                item.update(detail)
                results.append(WelfareRaw(**item))
            except (httpx.HTTPError, httpx.TimeoutException, ET.ParseError, KeyError, ValueError) as e:
                logger.warning("항목 스킵: %s", str(e))
                continue

        if len(items) < per_page:
            break

    return results

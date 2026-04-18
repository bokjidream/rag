"""공공데이터 복지서비스 API 클라이언트."""
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://apis.data.go.kr/B554287/NationalWelfareInformationsV001"
LIST_URL = f"{_BASE_URL}/NationalWelfarelistV001"
DETAIL_URL = f"{_BASE_URL}/NationalWelfaredetailedV001"
PAGE_SIZE = 100


async def fetch_welfare_list(
    client: httpx.AsyncClient,
    page: int = 1,
    num_rows: int = PAGE_SIZE,
) -> dict[str, Any]:
    """복지서비스 목록 API 호출 — 한 페이지."""
    params = {
        "serviceKey": os.environ["WELFARE_API_KEY"],
        "callTp": "R",
        "pageNo": str(page),
        "numOfRows": str(num_rows),
    }
    response = await client.get(LIST_URL, params=params)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


async def fetch_welfare_detail(
    client: httpx.AsyncClient,
    serv_id: str,
) -> dict[str, Any]:
    """복지서비스 상세 API 호출."""
    params = {
        "serviceKey": os.environ["WELFARE_API_KEY"],
        "callTp": "R",
        "servId": serv_id,
    }
    response = await client.get(DETAIL_URL, params=params)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]

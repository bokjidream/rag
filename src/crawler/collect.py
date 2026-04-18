"""복지서비스 전체 수집 — 공공데이터 API → data/raw/welfare_services.json."""
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from src.crawler.client import PAGE_SIZE, fetch_welfare_detail, fetch_welfare_list

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("data/raw/welfare_services.json")


async def _collect_serv_ids(client: httpx.AsyncClient) -> list[str]:
    """목록 API 페이지네이션으로 전체 servId 수집."""
    serv_ids: list[str] = []
    page = 1

    while True:
        data = await fetch_welfare_list(client, page=page, num_rows=PAGE_SIZE)
        wanted = data.get("wantedList", {})
        items: list[dict[str, Any]] = wanted.get("servList", [])

        if not items:
            break

        for item in items:
            serv_id = item.get("servId")
            if serv_id:
                serv_ids.append(str(serv_id))

        total = int(wanted.get("totCnt", 0))
        if len(serv_ids) >= total:
            break

        page += 1

    logger.info("servId 수집 완료: %d개", len(serv_ids))
    return serv_ids


async def collect_all(
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    """복지서비스 전체 수집 후 JSON 저장.

    실패한 servId는 skip하고 나머지를 계속 수집한다.
    """
    services: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        serv_ids = await _collect_serv_ids(client)

        for serv_id in serv_ids:
            try:
                data = await fetch_welfare_detail(client, serv_id)
                detail = data.get("wantedDtl", {})
                if detail:
                    services.append(detail)
            except Exception as exc:
                logger.warning("상세 수집 실패, 건너뜀: servId=%s error=%s", serv_id, exc)
                continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(services, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("수집 완료: %d개 → %s", len(services), output_path)
    return services

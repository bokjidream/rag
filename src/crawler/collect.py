from __future__ import annotations

import asyncio
import logging
import os
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from playwright.async_api import async_playwright

from src.crawler.welfare_detail_web import _UA, scrape_welfare_detail
from src.crawler.welfare_list import fetch_welfare_list
from src.models.welfare import WelfareRaw

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 5  # 동시 Playwright 페이지 수


async def collect_all(max_pages: int = 10, per_page: int = 100) -> list[WelfareRaw]:
    api_key = os.environ.get("PUBLIC_DATA_API_KEY")
    if not api_key:
        raise ValueError("PUBLIC_DATA_API_KEY 환경변수가 설정되지 않았습니다.")

    # Phase 1: 목록 수집 (API — 할당량 적음)
    all_items: list[dict[str, Any]] = []
    for page_num in range(1, max_pages + 1):
        try:
            items = await fetch_welfare_list(api_key=api_key, page=page_num, per_page=per_page)
        except (httpx.HTTPError, httpx.TimeoutException, ET.ParseError, KeyError, ValueError) as e:
            logger.warning("목록 페이지 %d 스킵: %s", page_num, str(e))
            break
        all_items.extend(items)
        if len(items) < per_page:
            break

    logger.info("목록 수집 완료: %d건 — 복지로 상세 크롤링 시작", len(all_items))

    # Phase 2: 상세 크롤링 (복지로 웹, Playwright)
    results = await _crawl_details(all_items)
    return results


async def _crawl_details(items: list[dict[str, Any]]) -> list[WelfareRaw]:
    if not items:
        return []

    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    def _to_welfare_raw(item: dict[str, Any]) -> WelfareRaw | None:
        serv_id = item.get("serv_id", "")
        try:
            return WelfareRaw(**item)
        except (TypeError, ValueError) as e:
            logger.warning("모델 오류 스킵 (%s): %s", serv_id, e)
            return None

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=_UA)

            async def process(item: dict[str, Any], idx: int) -> WelfareRaw | None:
                serv_id = item.get("serv_id", "")
                url = item.get("serv_dtl_link", "")
                if not url:
                    logger.warning("상세 링크 없음 (%s)", serv_id)
                    return _to_welfare_raw(item)

                merged: dict[str, Any] = item
                async with sem:
                    page = await context.new_page()
                    try:
                        detail = await scrape_welfare_detail(url, page)
                        merged = {**item, **detail}
                        if (idx + 1) % 50 == 0:
                            logger.info("크롤링 진행: %d/%d", idx + 1, len(items))
                    except Exception as e:
                        logger.warning("크롤링 실패 (%s): %s", serv_id, e)
                    finally:
                        await page.close()

                return _to_welfare_raw(merged)

            tasks = [process(item, idx) for idx, item in enumerate(items)]
            try:
                done = await asyncio.gather(*tasks)
            finally:
                await browser.close()
    except Exception as e:
        logger.warning("브라우저 크롤러 초기화 실패, 목록 데이터만 반환: %s", e)
        done = [_to_welfare_raw(item) for item in items]

    results = [r for r in done if r is not None]
    logger.info("크롤링 완료: %d건 수집", len(results))
    return results

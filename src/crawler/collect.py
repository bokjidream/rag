from __future__ import annotations

import asyncio
import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from playwright.async_api import async_playwright

from src.crawler.client import build_client
from src.crawler.welfare_detail import fetch_welfare_detail
from src.crawler.welfare_detail_web import _UA, scrape_welfare_detail
from src.crawler.welfare_list import fetch_welfare_list
from src.models.welfare import WelfareRaw

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 5  # 동시 Playwright 페이지 수
_APPLICATION_FORMS_API_LIMIT = "APPLICATION_FORMS_API_LIMIT"
_APPLICATION_FORMS_STATE_PATH = "APPLICATION_FORMS_STATE_PATH"
_DEFAULT_APPLICATION_FORMS_STATE_PATH = Path("data/processed/application_forms_fetch_state.json")
_APPLICATION_FORMS_FETCH_DELAY_SECONDS = 0.4


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

    logger.info("목록 수집 완료: %d건 — 신청서 자료 상세 API 보강 시작", len(all_items))

    # Phase 2: 신청서 자료 보강 (상세 API basfrmList, 기본값은 quota 보호를 위해 0건)
    all_items = await _enrich_application_forms(all_items, api_key)

    logger.info("신청서 자료 보강 완료 — 복지로 상세 크롤링 시작")

    # Phase 3: 상세 크롤링 (복지로 웹, Playwright)
    results = await _crawl_details(all_items)
    return results


def _application_forms_api_limit() -> int:
    raw = os.environ.get(_APPLICATION_FORMS_API_LIMIT, "0")
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("%s 값이 정수가 아니어서 0으로 처리합니다: %s", _APPLICATION_FORMS_API_LIMIT, raw)
        return 0


def _application_forms_state_path() -> Path:
    raw = os.environ.get(_APPLICATION_FORMS_STATE_PATH)
    if raw:
        return Path(raw)
    return _DEFAULT_APPLICATION_FORMS_STATE_PATH


def _normalize_application_forms(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    def _clean_text(raw: Any) -> str:
        if raw is None:
            return ""
        return str(raw).strip()

    forms: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        title = _clean_text(item.get("title"))
        url = _clean_text(item.get("url"))
        file_type = _clean_text(item.get("file_type")).lower() or "etc"
        if file_type not in {"pdf", "hwp", "hwpx", "etc"}:
            file_type = "etc"

        if title or url:
            forms.append({"title": title, "url": url, "file_type": file_type})

    return forms


def _load_application_forms_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("신청서 자료 fetch 상태 파일을 읽지 못했습니다 (%s): %s", path, e)
        return {}

    if not isinstance(raw, dict):
        logger.warning("신청서 자료 fetch 상태 파일 형식이 올바르지 않습니다: %s", path)
        return {}

    state: dict[str, dict[str, Any]] = {}
    for serv_id, entry in raw.items():
        if isinstance(serv_id, str) and isinstance(entry, dict):
            state[serv_id] = dict(entry)
    return state


def _write_application_forms_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning("신청서 자료 fetch 상태 파일을 저장하지 못했습니다 (%s): %s", path, e)


def _apply_cached_application_forms(
    items: list[dict[str, Any]],
    state: dict[str, dict[str, Any]],
) -> None:
    for item in items:
        serv_id = str(item.get("serv_id", ""))
        entry = state.get(serv_id)
        if entry is None or entry.get("status") != "success":
            continue
        item["application_forms"] = _normalize_application_forms(entry.get("application_forms", []))


async def _enrich_application_forms(
    items: list[dict[str, Any]],
    api_key: str,
) -> list[dict[str, Any]]:
    if not items:
        return []

    enriched = [dict(item) for item in items]
    state_path = _application_forms_state_path()
    state = _load_application_forms_state(state_path)
    _apply_cached_application_forms(enriched, state)

    limit = _application_forms_api_limit()
    if limit <= 0:
        logger.info("신청서 자료 상세 API 호출 스킵: %s=0", _APPLICATION_FORMS_API_LIMIT)
        return enriched

    fetched = 0
    changed = False
    async with build_client() as client:
        for item in enriched:
            if fetched >= limit:
                break

            serv_id = str(item.get("serv_id", ""))
            if not serv_id:
                continue
            if state.get(serv_id, {}).get("status") == "success":
                continue

            try:
                detail = await fetch_welfare_detail(serv_id, api_key=api_key, client=client)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("상세 API 일일/분당 호출 제한 도달: %s", e)
                    break
                logger.warning("신청서 자료 상세 API 호출 실패 (%s): %s", serv_id, e)
                continue
            except (httpx.HTTPError, ET.ParseError, ValueError) as e:
                logger.warning("신청서 자료 상세 API 호출 실패 (%s): %s", serv_id, e)
                continue

            forms = _normalize_application_forms(detail.get("application_forms", []))
            item["application_forms"] = forms
            state[serv_id] = {
                "status": "success",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "application_forms": forms,
            }
            fetched += 1
            changed = True

            if fetched < limit:
                await asyncio.sleep(_APPLICATION_FORMS_FETCH_DELAY_SECONDS)

    if changed:
        _write_application_forms_state(state_path, state)

    logger.info("신청서 자료 상세 API 호출 완료: %d건", fetched)
    return enriched


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

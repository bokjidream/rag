from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from playwright.async_api import Page

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_NAV_TIMEOUT = 30_000
_RENDER_MS = 2_000
_SECTION_SELECTORS = (".cl-container", ".cl-htmlsnippet", ".cl-output")
_SECTION_HEADERS = ("지원대상", "선정기준", "서비스 내용", "신청방법", "추가정보")


def _is_content(text: str) -> bool:
    if len(text) < 30:
        return False
    skip_prefixes = ("div.tooltip", ".cl-", ".new-sf", "업무별")
    if any(text.startswith(p) for p in skip_prefixes):
        return False
    if "전자정부" in text[:60]:
        return False
    if "{" in text[:80] and "}" in text[:80] and ":" in text[:80]:
        return False
    return True


async def _content_snippets(page: Page) -> list[str]:
    els = await page.query_selector_all(".cl-htmlsnippet")
    result = []
    for el in els:
        text = (await el.inner_text()).strip()
        if _is_content(text):
            result.append(text)
    return result


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalized_header(line: str) -> str:
    return re.sub(r"\s+", "", line)


def _header_variants(label: str) -> set[str]:
    normalized = _normalized_header(label)
    return {
        normalized,
        normalized * 2,
        f"{normalized}선택됨",
        f"{normalized}선택됨{normalized}",
    }


def _is_section_header(line: str, label: str) -> bool:
    return _normalized_header(line) in _header_variants(label)


def _extract_section_from_text(text: str, label: str) -> str:
    text = _normalize_text(text)
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start_idx = -1
    for idx, line in enumerate(lines):
        if _is_section_header(line, label):
            start_idx = idx
            break
    if start_idx == -1:
        return ""

    section_lines = lines[start_idx:]
    while section_lines and _is_section_header(section_lines[0], label):
        section_lines.pop(0)

    if not section_lines:
        return ""

    result_lines: list[str] = []
    for line in section_lines:
        if any(_is_section_header(line, header) for header in _SECTION_HEADERS if header != label):
            break
        result_lines.append(line)

    result = "\n".join(result_lines).strip()
    if not result:
        return ""
    if result.startswith(label):
        result = result[len(label) :].strip()
    return result


async def _extract_section(page: Page, label: str) -> str:
    candidates: list[str] = []

    for selector in _SECTION_SELECTORS:
        for element in await page.query_selector_all(selector):
            try:
                text = await element.inner_text()
            except Exception:
                continue

            section = _extract_section_from_text(text, label)
            if section:
                candidates.append(section)

    if not candidates:
        return ""

    # 페이지 전체를 긁어온 상위 컨테이너보다 섹션 단위의 짧은 컨테이너를 우선한다.
    return min(candidates, key=len)


async def _click_service_content_tab(page: Page) -> None:
    tab = page.locator("a", has_text="서비스 내용").first
    if await tab.count() == 0:
        return

    await tab.click(force=True)
    await page.wait_for_timeout(_RENDER_MS)


async def scrape_welfare_detail(url: str, page: Page) -> dict[str, Any]:
    """복지로 상세 페이지를 Playwright로 크롤링해 상세 텍스트를 반환한다.

    Returns:
        tgtr_dtl_cn, slct_crit_cn, alw_serv_cn 필드를 담은 dict.
        실패 시 빈 문자열로 채워 반환한다.
    """
    await page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT)
    await asyncio.sleep(_RENDER_MS / 1000)

    tgtr_dtl_cn = await _extract_section(page, "지원대상")
    slct_crit_cn = await _extract_section(page, "선정기준")

    alw_serv_cn = ""
    try:
        await _click_service_content_tab(page)
        alw_serv_cn = await _extract_section(page, "서비스 내용")
    except Exception as e:
        logger.debug("서비스 내용 탭 스킵 (%s): %s", url, e)

    if not (tgtr_dtl_cn and slct_crit_cn and alw_serv_cn):
        snippets = await _content_snippets(page)
        if not tgtr_dtl_cn and len(snippets) > 1:
            tgtr_dtl_cn = snippets[1]
        if not slct_crit_cn and len(snippets) > 2:
            slct_crit_cn = snippets[2]
        if not alw_serv_cn and len(snippets) > 3:
            alw_serv_cn = snippets[3]

    return {
        "tgtr_dtl_cn": tgtr_dtl_cn,
        "slct_crit_cn": slct_crit_cn,
        "alw_serv_cn": alw_serv_cn,
    }

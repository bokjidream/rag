from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_NAV_TIMEOUT = 30_000
_RENDER_MS = 2_000
_TAB_TIMEOUT = 5_000
_SECTION_SELECTORS = (".cl-container", ".cl-htmlsnippet", ".cl-output")
_SECTION_HEADERS = ("지원대상", "선정기준", "서비스 내용", "신청방법", "추가정보")

_DOC_HEADING_RE = re.compile(
    r"^[※\*\-\s]*(구비서류|제출서류|필요서류|첨부서류|신청서류)"
    r"\s*(?:는|은)?\s*(?:다음과 같습니다|입니다)?\.?\s*$"
)
_BULLET_RE = re.compile(r"^(?:[○●▶▷·•\-]|[①-⑳]|\d+[.)])\s*(.+)$")
_INLINE_SPLIT_RE = re.compile(r"[,，、]")
_EMPTY_VALUE_RE = re.compile(r"^(없음|해당\s*없음|해당사항\s*없음|없습니다|별도\s*없음|해당없음)$")


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


def _parse_required_documents(text: str) -> list[str]:
    """신청방법 원문에서 구비/제출/첨부 서류 항목을 추출한다.

    키워드가 없거나 패턴이 불분명하면 빈 리스트를 반환한다.
    잘못된 서류 안내가 누락보다 위험하므로 애매하면 빈 배열을 반환한다.
    """
    raw_lines = [raw.strip() for raw in text.splitlines() if raw.strip()]
    docs: list[str] = []

    def _is_valid(item: str) -> bool:
        return bool(item) and not _EMPTY_VALUE_RE.match(item) and not _DOC_HEADING_RE.match(item)

    def _append_inline(value: str) -> None:
        docs.extend(x.strip() for x in _INLINE_SPLIT_RE.split(value) if _is_valid(x.strip()))

    def _normalize_list_item(item: str) -> str:
        colon_parts = re.split(r"[:：]", item, maxsplit=1)
        if len(colon_parts) == 2:
            return colon_parts[1].strip()
        return item

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        colon_parts = re.split(r"[:：]", line, maxsplit=1)
        if len(colon_parts) == 2 and _DOC_HEADING_RE.match(colon_parts[0].strip()):
            after = colon_parts[1].strip()
            if after:
                _append_inline(after)
                i += 1
                continue

        if _DOC_HEADING_RE.match(line):
            i += 1
            while i < len(raw_lines):
                m = _BULLET_RE.match(raw_lines[i])
                if m:
                    item = _normalize_list_item(m.group(1).strip())
                    if _is_valid(item):
                        docs.append(item)
                    i += 1
                else:
                    break
            continue
        i += 1

    return docs


async def _click_tab(page: Page, label: str) -> None:
    tab = page.locator("a", has_text=label).first
    try:
        await tab.wait_for(state="attached", timeout=_TAB_TIMEOUT)
    except PlaywrightTimeoutError:
        return

    await tab.click(force=True)
    await page.wait_for_timeout(_RENDER_MS)


async def _click_service_content_tab(page: Page) -> None:
    await _click_tab(page, "서비스 내용")


async def _click_application_method_tab(page: Page) -> None:
    await _click_tab(page, "신청방법")


async def scrape_welfare_detail(url: str, page: Page) -> dict[str, Any]:
    """복지로 상세 페이지를 Playwright로 크롤링해 상세 텍스트를 반환한다.

    Returns:
        tgtr_dtl_cn, slct_crit_cn, alw_serv_cn, application_method,
        required_documents 필드를 담은 dict.
        실패 시 빈 문자열/빈 리스트로 채워 반환한다.
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

    application_method = ""
    try:
        await _click_application_method_tab(page)
        application_method = await _extract_section(page, "신청방법")
    except Exception as e:
        logger.debug("신청방법 탭 스킵 (%s): %s", url, e)

    if not (tgtr_dtl_cn and slct_crit_cn and alw_serv_cn):
        snippets = await _content_snippets(page)
        if not tgtr_dtl_cn and len(snippets) > 1:
            tgtr_dtl_cn = snippets[1]
        if not slct_crit_cn and len(snippets) > 2:
            slct_crit_cn = snippets[2]
        if not alw_serv_cn and len(snippets) > 3:
            alw_serv_cn = snippets[3]

    required_documents = _parse_required_documents(application_method) if application_method else []

    return {
        "tgtr_dtl_cn": tgtr_dtl_cn,
        "slct_crit_cn": slct_crit_cn,
        "alw_serv_cn": alw_serv_cn,
        "application_method": application_method,
        "required_documents": required_documents,
    }

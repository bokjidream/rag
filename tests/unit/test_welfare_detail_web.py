from __future__ import annotations

from src.crawler.welfare_detail_web import (
    _extract_section_from_text,
    _is_content,
)

# ── _is_content ──────────────────────────────────────────────────────────────


def test_is_content_returns_false_for_short_text() -> None:
    assert _is_content("짧은 텍스트") is False


def test_is_content_returns_false_for_skip_prefix() -> None:
    assert _is_content("div.tooltip 이 텍스트는 툴팁을 설명하는 긴 내용입니다.") is False
    assert _is_content(".cl-container 이 텍스트는 클래스를 설명하는 긴 내용입니다.") is False
    assert _is_content(".new-sf 이 텍스트는 새 스타일을 설명하는 긴 내용입니다.") is False
    assert _is_content("업무별 담당자 정보가 여기에 길게 나오는 텍스트 내용입니다.") is False


def test_is_content_returns_false_for_jungbo_government() -> None:
    assert _is_content("전자정부 법령에 의거하여 개인정보를 처리하는 내용이 여기에 표시됩니다.") is False


def test_is_content_returns_false_for_json_like_content() -> None:
    assert _is_content('{"key": "value", "data": "info"} 이 텍스트는 JSON 형식의 내용입니다.') is False


def test_is_content_returns_true_for_valid_content() -> None:
    assert _is_content("보육교사 및 교사겸직원장의 근로여건 개선을 위해 근무환경개선비를 지원합니다.") is True


# ── _extract_section_from_text ────────────────────────────────────────────────


def test_extract_section_from_text_empty_input_returns_empty() -> None:
    assert _extract_section_from_text("", "지원대상") == ""
    assert _extract_section_from_text("   \n  \n  ", "지원대상") == ""


def test_extract_section_from_text_label_not_found_returns_empty() -> None:
    text = """
    신청방법
    방문 신청
    """
    assert _extract_section_from_text(text, "지원대상") == ""


def test_extract_section_from_text_header_only_returns_empty() -> None:
    text = """
    지원대상
    선정기준
    선정기준 내용입니다.
    """
    assert _extract_section_from_text(text, "지원대상") == ""


def test_extract_section_from_text_support_target() -> None:
    text = """
    지원대상
    교사근무환경개선비 지원대상은 다음과 같습니다.
    (지원조건)
    - 담임교사
    선정기준
    지원대상의 내용을 참고해주시기 바랍니다.
    """

    result = _extract_section_from_text(text, "지원대상")

    assert result == "교사근무환경개선비 지원대상은 다음과 같습니다.\n(지원조건)\n- 담임교사"


def test_extract_section_from_text_selection_criteria() -> None:
    text = """
    지원대상지원대상
    부모 등 보호자 및 자녀 누구나 이용 가능합니다.
    선정기준
    지원대상의 내용을 참고해 주시기 바랍니다.
    """

    result = _extract_section_from_text(text, "선정기준")

    assert result == "지원대상의 내용을 참고해 주시기 바랍니다."


def test_extract_section_from_text_service_content_with_selected_tab() -> None:
    text = """
    서비스 내용 선택됨
    서비스 내용
    보육교사 및 교사겸직원장의 근로여건 개선을 위해 근무환경개선비를 지원합니다.
    - 월 28만원
    신청방법
    방문 신청
    """

    result = _extract_section_from_text(text, "서비스 내용")

    assert result == "보육교사 및 교사겸직원장의 근로여건 개선을 위해 근무환경개선비를 지원합니다.\n- 월 28만원"

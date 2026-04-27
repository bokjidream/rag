from __future__ import annotations

from src.crawler.welfare_detail_web import _extract_section_from_text


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

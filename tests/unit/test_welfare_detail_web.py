from __future__ import annotations

from src.crawler.welfare_detail_web import (
    _extract_section_from_text,
    _is_content,
    _parse_required_documents,
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
    assert (
        _is_content("전자정부 법령에 의거하여 개인정보를 처리하는 내용이 여기에 표시됩니다.")
        is False
    )


def test_is_content_returns_false_for_json_like_content() -> None:
    assert (
        _is_content('{"key": "value", "data": "info"} 이 텍스트는 JSON 형식의 내용입니다.') is False
    )


def test_is_content_returns_true_for_valid_content() -> None:
    assert (
        _is_content("보육교사 및 교사겸직원장의 근로여건 개선을 위해 근무환경개선비를 지원합니다.")
        is True
    )


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


# ── _parse_required_documents ────────────────────────────────────────────────


def test_parse_required_documents_inline_pattern() -> None:
    text = "주민센터 방문 신청\n구비서류: 신분증, 가족관계증명서\n처리기간: 14일"
    assert _parse_required_documents(text) == ["신분증", "가족관계증명서"]


def test_parse_required_documents_bullet_pattern() -> None:
    text = "신청방법\n주민센터 방문\n제출서류\n○ 신분증\n○ 진단서\n○ 소득확인서"
    assert _parse_required_documents(text) == ["신분증", "진단서", "소득확인서"]


def test_parse_required_documents_no_keyword_returns_empty() -> None:
    text = "읍면동 주민센터에서 신청 가능합니다. 온라인 신청 불가."
    assert _parse_required_documents(text) == []


def test_parse_required_documents_empty_string_returns_empty() -> None:
    assert _parse_required_documents("") == []


def test_parse_required_documents_filters_empty_values() -> None:
    text = "구비서류: 없음"
    assert _parse_required_documents(text) == []


def test_parse_required_documents_filters_empty_values_bullet() -> None:
    text = "첨부서류\n○ 해당 없음"
    assert _parse_required_documents(text) == []


def test_parse_required_documents_filters_various_empty_phrases() -> None:
    for phrase in ["없음", "해당 없음", "해당없음", "해당사항 없음", "없습니다", "별도 없음"]:
        text = f"구비서류: {phrase}"
        assert _parse_required_documents(text) == [], f"'{phrase}'는 빈 배열이어야 함"


def test_parse_required_documents_ignores_keyword_in_sentence() -> None:
    text = (
        "대상 영아의 부모가 출생일로부터 1년 이내에 제출서류를 구비하여 "
        "보건소로 신청 또는 e보건소 공공보건포털(http://www.e-health.go.kr), "
        "아이마중앱 등 온라인 신청이 가능합니다."
    )

    assert _parse_required_documents(text) == []


def test_parse_required_documents_numbered_items_after_heading() -> None:
    text = """
    신청방법 : 발급신청서 온라인 작성 및 제출서류를 스캔하여 파일 첨부
    제출서류
    ① 본인신청 : 해당사항 없음
    ② 가족신청 : 주민등록등본 또는 가족관계증명서 1부, 신분증 사본 각 1부
    ③ 대리신청 : 위임장
    """

    assert _parse_required_documents(text) == [
        "주민등록등본 또는 가족관계증명서 1부, 신분증 사본 각 1부",
        "위임장",
    ]


def test_parse_required_documents_star_heading_then_dash_items() -> None:
    text = """
    장학 신청기간내에 신청서와 구비서류를 준비하여 신청합니다.
    * 구비서류
    - 직전학기성적증명서 1부
    - 재학증명서 1부
    """

    assert _parse_required_documents(text) == ["직전학기성적증명서 1부", "재학증명서 1부"]


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

    assert (
        result
        == "보육교사 및 교사겸직원장의 근로여건 개선을 위해 근무환경개선비를 지원합니다.\n- 월 28만원"
    )


def test_extract_section_from_text_application_method_with_selected_tab() -> None:
    text = """
    신청방법 선택됨
    신청방법
    읍면동 주민센터 방문 신청
    구비서류: 신분증, 가족관계증명서
    추가정보
    문의처 정보
    """

    result = _extract_section_from_text(text, "신청방법")

    assert result == "읍면동 주민센터 방문 신청\n구비서류: 신분증, 가족관계증명서"

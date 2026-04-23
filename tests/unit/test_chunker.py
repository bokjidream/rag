from __future__ import annotations

import pytest

from src.models.welfare import WelfareRaw
from src.pipeline.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text, make_document_text


def _make_item(**kwargs: object) -> WelfareRaw:
    defaults: dict[str, object] = {
        "serv_id": "WLF001",
        "serv_nm": "테스트 서비스",
        "serv_dgst": "서비스 개요",
        "jur_mnof_nm": "보건복지부",
        "trgter_indvdl": ["저소득"],
        "intrs_thema": ["생활지원"],
        "sprt_cyc_nm": "월",
        "srv_pvsn_nm": "현금지급",
        "serv_dtl_link": "https://bokjiro.go.kr",
        "tgtr_dtl_cn": "수급 대상 상세",
        "slct_crit_cn": "선정 기준",
        "alw_serv_cn": "서비스 내용",
    }
    defaults.update(kwargs)
    return WelfareRaw(**defaults)  # type: ignore[arg-type]


# ── chunk_text ────────────────────────────────────────────────────────────────


def test_chunk_text_short_text_returns_single_chunk() -> None:
    text = "가" * (CHUNK_SIZE - 1)
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_exactly_chunk_size_returns_single_chunk() -> None:
    text = "가" * CHUNK_SIZE
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_text_returns_multiple_chunks() -> None:
    text = "나" * (CHUNK_SIZE * 2)
    chunks = chunk_text(text)
    assert len(chunks) > 1


def test_chunk_text_overlap_is_correct() -> None:
    text = "가" * (CHUNK_SIZE + 100)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    # chunk[0]의 마지막 CHUNK_OVERLAP 글자 == chunk[1]의 첫 CHUNK_OVERLAP 글자
    assert chunks[0][-CHUNK_OVERLAP:] == chunks[1][:CHUNK_OVERLAP]


def test_chunk_text_empty_string_returns_empty_list() -> None:
    assert chunk_text("") == []


def test_chunk_text_all_chunks_within_size() -> None:
    text = "다" * (CHUNK_SIZE * 3 + 50)
    chunks = chunk_text(text)
    for chunk in chunks:
        assert len(chunk) <= CHUNK_SIZE


# ── make_document_text ───────────────────────────────────────────────────────


def test_make_document_text_joins_fields_with_double_newline() -> None:
    item = _make_item(
        serv_nm="서비스명",
        serv_dgst="개요",
        tgtr_dtl_cn="대상",
        slct_crit_cn="기준",
        alw_serv_cn="내용",
    )
    text = make_document_text(item)
    parts = text.split("\n\n")
    assert "서비스명" in parts
    assert "개요" in parts
    assert "대상" in parts
    assert "기준" in parts
    assert "내용" in parts


def test_make_document_text_with_empty_optional_fields() -> None:
    item = _make_item(tgtr_dtl_cn="", slct_crit_cn="", alw_serv_cn="")
    text = make_document_text(item)
    # 빈 필드는 필터링되므로 연속 이중 개행이 없어야 함
    assert "\n\n\n\n" not in text
    assert "테스트 서비스" in text
    assert "서비스 개요" in text


def test_make_document_text_all_optional_empty_returns_only_required() -> None:
    item = _make_item(
        serv_nm="이름만",
        serv_dgst="",
        tgtr_dtl_cn="",
        slct_crit_cn="",
        alw_serv_cn="",
    )
    text = make_document_text(item)
    assert text == "이름만"


def test_make_document_text_field_order() -> None:
    """serv_nm이 텍스트 맨 앞에 위치해야 한다."""
    item = _make_item()
    text = make_document_text(item)
    assert text.startswith("테스트 서비스")

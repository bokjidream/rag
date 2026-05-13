from __future__ import annotations

import pytest

from src.models.welfare import WelfareRaw
from src.pipeline.chunker import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TOKEN_MAX,
    TOKEN_OVERLAP,
    WelfareChunk,
    _window_ranges,
    chunk_item,
    chunk_text,
    make_document_text,
)


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


class _FakeTokenizer:
    """테스트용 tokenizer. 숫자 토큰 문자열은 encode/decode 라운드트립을 보장."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        pieces = text.split()
        if pieces and all(p.isdigit() for p in pieces):
            ids = [int(p) for p in pieces]
        else:
            ids = list(range(len(text)))
        return [-101, *ids, -102] if add_special_tokens else ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        ids = [i for i in token_ids if i >= 0] if skip_special_tokens else token_ids
        return " ".join(str(i) for i in ids)


class _ExpandingDecodeTokenizer:
    """decode 후 재토큰화 길이가 늘어나는 tokenizer edge case."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = list(range(len(text)))
        return [-101, *ids, -102] if add_special_tokens else ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "X" * (len(token_ids) + 5)


class _UnknownTokenizer:
    """skip_special_tokens=True에서 [UNK]가 빈 문자열로 decode되는 edge case."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [3]
        return [-101, *ids, -102] if add_special_tokens else ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "" if skip_special_tokens else "[UNK]"


def _token_text(length: int) -> str:
    return " ".join(str(i) for i in range(length))


def _make_token_item(length: int) -> WelfareRaw:
    return _make_item(
        serv_nm=_token_text(length),
        serv_dgst="",
        tgtr_dtl_cn="",
        slct_crit_cn="",
        alw_serv_cn="",
    )


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


# ── chunk_item ────────────────────────────────────────────────────────────────


def test_window_ranges_exact_fit() -> None:
    assert _window_ranges(TOKEN_MAX, TOKEN_MAX, TOKEN_OVERLAP) == [(0, TOKEN_MAX)]


def test_window_ranges_no_tail_duplicate() -> None:
    stride = TOKEN_MAX - TOKEN_OVERLAP
    assert _window_ranges(TOKEN_MAX + stride, TOKEN_MAX, TOKEN_OVERLAP) == [
        (0, TOKEN_MAX),
        (stride, TOKEN_MAX + stride),
    ]


def test_window_ranges_three_windows() -> None:
    stride = TOKEN_MAX - TOKEN_OVERLAP
    assert _window_ranges(TOKEN_MAX + stride + 1, TOKEN_MAX, TOKEN_OVERLAP) == [
        (0, TOKEN_MAX),
        (stride, TOKEN_MAX + stride),
        (stride * 2, TOKEN_MAX + stride + 1),
    ]


@pytest.mark.parametrize(
    ("window_size", "overlap"),
    [(0, 0), (TOKEN_MAX, TOKEN_MAX), (TOKEN_MAX, TOKEN_MAX + 1), (TOKEN_MAX, -1)],
)
def test_window_ranges_invalid_args(window_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        _window_ranges(TOKEN_MAX, window_size, overlap)


def test_chunk_item_returns_at_least_one_chunk() -> None:
    item = _make_item()
    chunks = chunk_item(item)
    assert len(chunks) >= 1
    assert all(isinstance(c, WelfareChunk) for c in chunks)


def test_chunk_item_all_chunks_within_token_limit() -> None:
    tok = _FakeTokenizer()
    long_text = "보육교사 및 교사겸직원장의 근로여건 개선을 위해 근무환경개선비를 지원합니다. " * 20
    item = _make_item(tgtr_dtl_cn=long_text, slct_crit_cn=long_text)
    for chunk in chunk_item(item, tokenizer=tok):
        assert chunk.token_count <= TOKEN_MAX, f"TOKEN_MAX 초과: {chunk.token_count} > {TOKEN_MAX}"


def test_chunk_item_section_is_document() -> None:
    item = _make_item()
    chunks = chunk_item(item)
    assert {c.section for c in chunks} == {"document"}


def test_chunk_item_token_max_document_produces_one_chunk() -> None:
    tok = _FakeTokenizer()
    chunks = chunk_item(_make_token_item(TOKEN_MAX), tokenizer=tok)
    assert len(chunks) == 1
    assert chunks[0].token_count == TOKEN_MAX


def test_chunk_item_token_max_plus_stride_produces_two_chunks() -> None:
    tok = _FakeTokenizer()
    stride = TOKEN_MAX - TOKEN_OVERLAP
    chunks = chunk_item(_make_token_item(TOKEN_MAX + stride), tokenizer=tok)
    assert len(chunks) == 2
    assert [c.token_count for c in chunks] == [TOKEN_MAX, TOKEN_MAX]


def test_chunk_item_three_windows() -> None:
    tok = _FakeTokenizer()
    stride = TOKEN_MAX - TOKEN_OVERLAP
    chunks = chunk_item(_make_token_item(TOKEN_MAX + stride + 1), tokenizer=tok)
    assert len(chunks) == 3
    assert [c.token_count for c in chunks] == [TOKEN_MAX, TOKEN_MAX, TOKEN_OVERLAP + 1]


def test_chunk_item_overlap_between_consecutive_chunks() -> None:
    tok = _FakeTokenizer()
    chunks = chunk_item(_make_token_item(TOKEN_MAX + 1), tokenizer=tok)
    first_ids = tok.encode(chunks[0].text, add_special_tokens=False)
    second_ids = tok.encode(chunks[1].text, add_special_tokens=False)
    assert first_ids[-TOKEN_OVERLAP:] == second_ids[:TOKEN_OVERLAP]


def test_chunk_item_encode_decode_token_count_within_limit() -> None:
    tok = _FakeTokenizer()
    stride = TOKEN_MAX - TOKEN_OVERLAP
    chunks = chunk_item(_make_token_item(TOKEN_MAX + stride + 1), tokenizer=tok)
    for chunk in chunks:
        assert len(tok.encode(chunk.text, add_special_tokens=False)) <= TOKEN_MAX


def test_chunk_item_trims_decoded_text_to_token_limit() -> None:
    tok = _ExpandingDecodeTokenizer()
    item = _make_item(
        serv_nm="A" * TOKEN_MAX,
        serv_dgst="",
        tgtr_dtl_cn="",
        slct_crit_cn="",
        alw_serv_cn="",
    )
    chunks = chunk_item(item, tokenizer=tok)
    assert len(chunks) == 1
    assert chunks[0].token_count == TOKEN_MAX
    assert len(tok.encode(chunks[0].text, add_special_tokens=False)) == TOKEN_MAX


def test_chunk_item_preserves_unknown_token_text() -> None:
    tok = _UnknownTokenizer()
    item = _make_item(
        serv_nm="A",
        serv_dgst="",
        tgtr_dtl_cn="",
        slct_crit_cn="",
        alw_serv_cn="",
    )
    chunks = chunk_item(item, tokenizer=tok)
    assert len(chunks) == 1
    assert chunks[0].text == "[UNK]"
    assert chunks[0].token_count == 1


def test_chunk_item_empty_optional_fields_returns_single_chunk() -> None:
    item = _make_item(
        serv_nm="서비스명",
        serv_dgst="개요 설명.",
        tgtr_dtl_cn="",
        slct_crit_cn="",
        alw_serv_cn="",
    )
    chunks = chunk_item(item)
    assert len(chunks) == 1
    assert "서비스명" in chunks[0].text


def test_chunk_item_long_document_produces_multiple_chunks() -> None:
    tok = _FakeTokenizer()
    item = _make_token_item(TOKEN_MAX * 2 + 100)
    chunks = chunk_item(item, tokenizer=tok)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= TOKEN_MAX, f"분할 후에도 TOKEN_MAX 초과: {chunk.token_count}"


def test_chunk_item_no_tokenizer_long_sentence_split() -> None:
    # tokenizer=None 경로: char 기반 추정으로 긴 전체 문서를 분할한다.
    long_text = "가" * 900
    item = _make_item(
        serv_nm="서비스명",
        serv_dgst="",
        tgtr_dtl_cn=long_text,
        slct_crit_cn="",
        alw_serv_cn="",
    )
    chunks = chunk_item(item, tokenizer=None)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.section == "document"
        assert chunk.token_count <= TOKEN_MAX

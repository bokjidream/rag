from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from src.models.welfare import WelfareRaw

# ── 레거시 상수 (기존 테스트 호환) ────────────────────────────────────────────
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

# ── 토큰 인식 청킹 상수 ───────────────────────────────────────────────────────
TOKEN_MAX = 480  # 512 - 2 special - 30 buffer
TOKEN_OVERLAP = 70  # ~1~2 문장
CHAR_PER_TOKEN = 1.82  # tokenizer 없을 때 글자→토큰 추정 (실측 역산)

ChunkSection = Literal[
    "document",
    "summary",
    "target",
    "criteria",
    "benefit",
    "application",
    "documents",
]


@dataclass(frozen=True)
class WelfareChunk:
    text: str
    section: ChunkSection
    token_count: int


def make_document_text(item: WelfareRaw) -> str:
    """WelfareRaw의 주요 텍스트 필드를 하나의 문서로 조합."""
    return "\n\n".join(
        filter(
            None,
            [
                item.serv_nm,
                item.serv_dgst,
                item.tgtr_dtl_cn,
                item.slct_crit_cn,
                item.alw_serv_cn,
            ],
        )
    )


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _document_section_text(metadata: Mapping[str, Any]) -> str:
    lines: list[str] = []

    for item in _json_list(metadata.get("required_documents", "[]")):
        text = _clean_text(item)
        if text:
            lines.append(f"필요서류: {text}")

    for item in _json_list(metadata.get("application_forms", "[]")):
        if isinstance(item, Mapping):
            title = _clean_text(item.get("title"))
            url = _clean_text(item.get("url"))
            file_type = _clean_text(item.get("file_type"))
            parts = [part for part in (title, file_type, url) if part]
            if parts:
                lines.append("신청서식: " + " ".join(parts))
        else:
            text = _clean_text(item)
            if text:
                lines.append(f"신청서식: {text}")

    return "\n".join(lines)


def _section_text(serv_nm: str, label: str, body: str) -> str:
    if not body:
        return ""
    return "\n\n".join(part for part in (serv_nm, label, body) if part)


def make_section_texts_from_metadata(
    metadata: Mapping[str, Any],
) -> list[tuple[ChunkSection, str]]:
    """Baseline metadata를 복지서비스 필드별 section text로 복원."""
    serv_nm = _clean_text(metadata.get("serv_nm"))
    summary = "\n\n".join(
        filter(
            None,
            [
                serv_nm,
                _clean_text(metadata.get("serv_dgst")),
            ],
        )
    )
    section_inputs: list[tuple[ChunkSection, str]] = [
        ("summary", summary),
        ("target", _section_text(serv_nm, "지원대상", _clean_text(metadata.get("tgtr_dtl_cn")))),
        (
            "criteria",
            _section_text(serv_nm, "선정기준", _clean_text(metadata.get("slct_crit_cn"))),
        ),
        ("benefit", _section_text(serv_nm, "지원내용", _clean_text(metadata.get("alw_serv_cn")))),
        (
            "application",
            _section_text(serv_nm, "신청방법", _clean_text(metadata.get("application_method"))),
        ),
        ("documents", _section_text(serv_nm, "필요서류 및 신청서식", _document_section_text(metadata))),
    ]
    return [(section, text) for section, text in section_inputs if text]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """텍스트를 청크 리스트로 분할 (슬라이딩 윈도우, 글자 수 기준)."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _count_tokens(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        return len(_encode_tokens(text, tokenizer))
    return max(1, round(len(text) / CHAR_PER_TOKEN))


def _encode_tokens(text: str, tokenizer: Any) -> list[int]:
    """special token 없이 토큰화하되 긴 원문 경고는 청킹 내부에서 숨긴다."""
    try:
        return list(
            tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
                verbose=False,
            )
        )
    except TypeError:
        return list(tokenizer.encode(text, add_special_tokens=False))


def _window_ranges(length: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    """length개 항목을 window_size/overlap 슬라이딩 윈도우로 분할."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be >= 0 and < window_size")
    if length <= 0:
        return []

    stride = window_size - overlap
    ranges: list[tuple[int, int]] = []
    start = 0

    while start < length:
        end = min(start + window_size, length)
        ranges.append((start, end))
        if end == length:
            break
        start += stride

    return ranges


def _decode_token_window(
    window_ids: list[int],
    tokenizer: Any,
) -> tuple[str, int]:
    """토큰 ID 윈도우를 text로 복원하고 재토큰화 기준 TOKEN_MAX를 보장."""
    text: str = tokenizer.decode(window_ids, skip_special_tokens=True)
    if not text and window_ids:
        text = tokenizer.decode(window_ids, skip_special_tokens=False)
    token_count = _count_tokens(text, tokenizer)

    while token_count > TOKEN_MAX and window_ids:
        overflow = token_count - TOKEN_MAX
        del window_ids[-max(1, min(len(window_ids), overflow)) :]
        text = tokenizer.decode(window_ids, skip_special_tokens=True)
        if not text and window_ids:
            text = tokenizer.decode(window_ids, skip_special_tokens=False)
        token_count = _count_tokens(text, tokenizer)

    return text, token_count


def _split_section_text(
    text: str,
    section: ChunkSection,
    tokenizer: Any | None,
) -> list[WelfareChunk]:
    if tokenizer is not None:
        ids = _encode_tokens(text, tokenizer)
        chunks: list[WelfareChunk] = []
        for start, end in _window_ranges(len(ids), TOKEN_MAX, TOKEN_OVERLAP):
            window_ids = ids[start:end]
            chunk_text, token_count = _decode_token_window(window_ids, tokenizer)
            chunks.append(
                WelfareChunk(
                    text=chunk_text,
                    section=section,
                    token_count=token_count,
                )
            )
        return chunks or [
            WelfareChunk(
                text=text,
                section=section,
                token_count=_count_tokens(text, tokenizer),
            )
        ]

    char_max = int(TOKEN_MAX * CHAR_PER_TOKEN)
    char_overlap = int(TOKEN_OVERLAP * CHAR_PER_TOKEN)
    chunks = []
    for start, end in _window_ranges(len(text), char_max, char_overlap):
        chunk_text = text[start:end]
        chunks.append(
            WelfareChunk(
                text=chunk_text,
                section=section,
                token_count=_count_tokens(chunk_text, None),
            )
        )
    return chunks or [
        WelfareChunk(
            text=text,
            section=section,
            token_count=_count_tokens(text, None),
        )
    ]


def chunk_metadata_sections(
    metadata: Mapping[str, Any],
    tokenizer: Any | None = None,
) -> list[WelfareChunk]:
    """Baseline Chroma metadata를 section-aware 청크로 분할."""
    chunks: list[WelfareChunk] = []
    for section, text in make_section_texts_from_metadata(metadata):
        chunks.extend(_split_section_text(text, section, tokenizer))
    return chunks


def chunk_item(
    item: WelfareRaw,
    tokenizer: Any | None = None,
) -> list[WelfareChunk]:
    """WelfareRaw를 전체 문서 기준 토큰 인식 슬라이딩 윈도우로 청크 분할."""
    full_text = make_document_text(item)

    if tokenizer is not None:
        ids = _encode_tokens(full_text, tokenizer)
        chunks: list[WelfareChunk] = []
        for start, end in _window_ranges(len(ids), TOKEN_MAX, TOKEN_OVERLAP):
            window_ids = ids[start:end]
            text, token_count = _decode_token_window(window_ids, tokenizer)
            chunks.append(
                WelfareChunk(
                    text=text,
                    section="document",
                    token_count=token_count,
                )
            )
        return chunks or [
            WelfareChunk(
                text=full_text,
                section="document",
                token_count=_count_tokens(full_text, tokenizer),
            )
        ]

    char_max = int(TOKEN_MAX * CHAR_PER_TOKEN)
    char_overlap = int(TOKEN_OVERLAP * CHAR_PER_TOKEN)
    chunks = []
    for start, end in _window_ranges(len(full_text), char_max, char_overlap):
        text = full_text[start:end]
        chunks.append(
            WelfareChunk(
                text=text,
                section="document",
                token_count=_count_tokens(text, None),
            )
        )
    return chunks or [
        WelfareChunk(
            text=full_text,
            section="document",
            token_count=_count_tokens(full_text, None),
        )
    ]

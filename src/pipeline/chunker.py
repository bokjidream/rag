from __future__ import annotations

from src.models.welfare import WelfareRaw

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50


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

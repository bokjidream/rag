from __future__ import annotations

import asyncio
import json

import chromadb

from src.db.chroma import WELFARE_COLLECTION, get_collection
from src.embedding.protocol import EmbedderProtocol
from src.models.welfare import WelfareRaw
from src.pipeline.chunker import chunk_text, make_document_text


async def index_welfare_items(
    items: list[WelfareRaw],
    embedder: EmbedderProtocol,
) -> int:
    """WelfareRaw 리스트를 청크 분할 → 임베딩 → ChromaDB upsert.

    Returns:
        upsert된 청크 수
    """
    if not items:
        return 0

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict[str, str]] = []

    for item in items:
        text = make_document_text(item)
        chunks = chunk_text(text)
        meta: dict[str, str] = {
            "serv_id": item.serv_id,
            "serv_nm": item.serv_nm,
            "serv_dgst": item.serv_dgst,
            "jur_mnof_nm": item.jur_mnof_nm,
            "trgter_indvdl": json.dumps(item.trgter_indvdl, ensure_ascii=False),
            "intrs_thema": json.dumps(item.intrs_thema, ensure_ascii=False),
            "sprt_cyc_nm": item.sprt_cyc_nm,
            "srv_pvsn_nm": item.srv_pvsn_nm,
            "serv_dtl_link": item.serv_dtl_link,
            "tgtr_dtl_cn": item.tgtr_dtl_cn,
            "slct_crit_cn": item.slct_crit_cn,
            "alw_serv_cn": item.alw_serv_cn,
        }
        for n, chunk in enumerate(chunks):
            all_ids.append(f"{item.serv_id}_chunk_{n}")
            all_docs.append(chunk)
            all_metas.append(meta)

    all_embeds: list[list[float]] = embedder.embed(all_docs)

    collection: chromadb.Collection = await get_collection(WELFARE_COLLECTION)

    # chromadb의 embeddings/metadatas 타입은 numpy array 기반 Union이라
    # list[list[float]]와 dict[str, str]를 직접 넘기면 mypy가 불일치를 보고한다.
    # 제로-인자 클로저로 감싸서 to_thread 호출 시 타입 검사를 우회한다.
    def _upsert() -> None:
        collection.upsert(
            ids=all_ids,
            documents=all_docs,
            embeddings=all_embeds,  # type: ignore[arg-type]
            metadatas=all_metas,  # type: ignore[arg-type]
        )

    await asyncio.to_thread(_upsert)

    return len(all_ids)

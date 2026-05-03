from __future__ import annotations

import asyncio
from typing import Any

import chromadb

from src.db.chroma import WELFARE_COLLECTION, get_collection
from src.embedding.protocol import EmbedderProtocol
from src.models.welfare import WelfareRaw
from src.pipeline.chunker import WelfareChunk, chunk_item


async def index_welfare_items(
    items: list[WelfareRaw],
    embedder: EmbedderProtocol,
    tokenizer: Any | None = None,
) -> int:
    """WelfareRaw 리스트를 청크 분할 → 임베딩 → ChromaDB upsert.

    Returns:
        upsert된 청크 수
    """
    if not items:
        return 0

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict[str, str | int]] = []

    for item in items:
        chunks: list[WelfareChunk] = chunk_item(item, tokenizer=tokenizer)
        meta = item.to_metadata()
        for n, chunk in enumerate(chunks):
            all_ids.append(f"{item.serv_id}_chunk_{n}")
            all_docs.append(chunk.text)
            all_metas.append(
                {
                    **meta,
                    "chunk_section": chunk.section,
                    "chunk_token_count": chunk.token_count,
                }
            )

    all_embeds: list[list[float]] = embedder.embed(all_docs)

    collection: chromadb.Collection = await get_collection(WELFARE_COLLECTION)

    def _get_existing_ids() -> list[str]:
        raw = collection.get(include=[])
        ids = raw.get("ids", [])
        return list(ids)

    existing_ids = await asyncio.to_thread(_get_existing_ids)
    next_id_set = set(all_ids)
    stale_ids = sorted(item_id for item_id in existing_ids if item_id not in next_id_set)

    if stale_ids:
        await asyncio.to_thread(collection.delete, ids=stale_ids)

    # chromadb의 embeddings/metadatas 타입은 numpy array 기반 Union이라
    # list[list[float]]와 dict[str, str | int]를 직접 넘기면 mypy가 불일치를 보고한다.
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

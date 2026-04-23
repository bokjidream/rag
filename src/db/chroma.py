from __future__ import annotations

import asyncio
import os

import chromadb
from chromadb.api import ClientAPI

WELFARE_COLLECTION = "welfare_services"

_client: ClientAPI | None = None
_lock: asyncio.Lock | None = None  # 이벤트 루프 실행 중에 생성해야 함 (Python 3.9 호환)


async def get_client() -> ClientAPI:
    """ChromaDB 클라이언트 싱글턴 반환.

    CHROMA_MODE=ephemeral  → EphemeralClient (in-memory, 테스트용)
    CHROMA_MODE=persistent → PersistentClient (기본값, data/chroma 경로)

    asyncio.Lock은 이벤트 루프 실행 중에 생성해야 한다 (Python 3.9).
    모듈 레벨에 두면 "attached to a different loop" RuntimeError 발생.
    Lock 없이 동시 호출 시 EphemeralClient가 중복 생성되어
    인덱싱 데이터가 검색에서 보이지 않는 버그가 발생한다.
    """
    global _client, _lock
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        if _client is None:
            mode = os.getenv("CHROMA_MODE", "persistent")
            if mode == "ephemeral":
                _client = await asyncio.to_thread(chromadb.EphemeralClient)
            else:
                path = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
                _client = await asyncio.to_thread(chromadb.PersistentClient, path)
    return _client


async def get_collection(name: str) -> chromadb.Collection:
    """컬렉션 반환. 없으면 생성. 거리 메트릭은 cosine으로 고정.

    retriever의 score = max(0, 1 - distance) 공식에 의존하므로
    hnsw:space=cosine을 변경하면 검색 점수가 깨진다.
    """
    client = await get_client()
    return await asyncio.to_thread(
        client.get_or_create_collection,
        name,
        metadata={"hnsw:space": "cosine"},
    )

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.db.chroma as db_module
from src.db.chroma import (
    WELFARE_COLLECTION,
    ChromaCollectionValidationError,
    get_client,
    get_collection,
    get_existing_collection,
    validate_existing_collection,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """각 테스트 전후로 싱글턴 상태를 초기화한다."""
    db_module._client = None
    db_module._lock = None
    yield
    db_module._client = None
    db_module._lock = None


@pytest.mark.asyncio
async def test_get_client_returns_same_instance_on_second_call():
    """get_client() 두 번 호출 시 동일 인스턴스를 반환해야 한다 (싱글턴)."""
    os.environ["CHROMA_MODE"] = "ephemeral"
    client1 = await get_client()
    client2 = await get_client()
    assert client1 is client2


@pytest.mark.asyncio
async def test_get_client_no_race_condition_on_concurrent_calls():
    """asyncio.gather로 동시 호출 시 동일 인스턴스를 반환해야 한다 (race condition 없음)."""
    os.environ["CHROMA_MODE"] = "ephemeral"
    clients = await __import__("asyncio").gather(get_client(), get_client(), get_client())
    assert clients[0] is clients[1]
    assert clients[1] is clients[2]


@pytest.mark.asyncio
async def test_get_client_ephemeral_mode():
    """CHROMA_MODE=ephemeral 시 EphemeralClient를 사용해야 한다."""
    os.environ["CHROMA_MODE"] = "ephemeral"
    client = await get_client()
    # EphemeralClient는 in-memory이므로 컬렉션 생성이 가능해야 한다
    assert client is not None
    # chromadb ClientAPI 인터페이스를 가져야 한다
    assert hasattr(client, "get_or_create_collection")


@pytest.mark.asyncio
async def test_get_client_persistent_mode(tmp_path):
    """CHROMA_MODE=persistent 시 PersistentClient를 사용해야 한다."""
    os.environ["CHROMA_MODE"] = "persistent"
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path)
    client = await get_client()
    assert client is not None
    assert hasattr(client, "get_or_create_collection")
    # 데이터 디렉토리가 생성되어야 한다
    assert tmp_path.exists()


@pytest.mark.asyncio
async def test_get_collection_returns_cosine_metric():
    """get_collection() 호출 시 hnsw:space=cosine 메타데이터로 컬렉션을 반환해야 한다."""
    os.environ["CHROMA_MODE"] = "ephemeral"
    collection = await get_collection("test_collection")
    assert collection is not None
    assert collection.metadata is not None
    assert collection.metadata.get("hnsw:space") == "cosine"


@pytest.mark.asyncio
async def test_welfare_collection_constant():
    """WELFARE_COLLECTION 상수가 올바르게 정의되어야 한다."""
    assert WELFARE_COLLECTION == "welfare_services"


@pytest.mark.asyncio
async def test_get_collection_welfare_default():
    """WELFARE_COLLECTION 이름으로 컬렉션을 가져올 수 있어야 한다."""
    os.environ["CHROMA_MODE"] = "ephemeral"
    collection = await get_collection(WELFARE_COLLECTION)
    assert collection.name == WELFARE_COLLECTION


@pytest.mark.asyncio
async def test_get_existing_collection_does_not_create(monkeypatch):
    """API startup 검증은 컬렉션을 생성하지 않아야 한다."""
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_client.get_or_create_collection = MagicMock()
    monkeypatch.setattr(db_module, "get_client", AsyncMock(return_value=mock_client))

    collection = await get_existing_collection("existing_collection")

    assert collection is mock_collection
    mock_client.get_collection.assert_called_once_with("existing_collection")
    mock_client.get_or_create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_validate_existing_collection_rejects_empty_collection(monkeypatch):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    monkeypatch.setattr(
        db_module,
        "get_existing_collection",
        AsyncMock(return_value=mock_collection),
    )

    with pytest.raises(ChromaCollectionValidationError, match="empty"):
        await validate_existing_collection("empty_collection")


@pytest.mark.asyncio
async def test_validate_existing_collection_requires_section_metadata(monkeypatch):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.get.return_value = {"metadatas": [{"serv_id": "WLF001"}]}
    monkeypatch.setattr(
        db_module,
        "get_existing_collection",
        AsyncMock(return_value=mock_collection),
    )

    with pytest.raises(ChromaCollectionValidationError, match="chunk_section"):
        await validate_existing_collection(
            "welfare_services_section_aware",
            require_section_metadata=True,
        )


@pytest.mark.asyncio
async def test_validate_existing_collection_rejects_partial_section_metadata(monkeypatch):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.get.return_value = {
        "metadatas": [
            {"chunk_section": "target"},
            {"serv_id": "WLF002"},
        ]
    }
    monkeypatch.setattr(
        db_module,
        "get_existing_collection",
        AsyncMock(return_value=mock_collection),
    )

    with pytest.raises(ChromaCollectionValidationError, match="chunk_section"):
        await validate_existing_collection(
            "welfare_services_section_aware",
            require_section_metadata=True,
        )


@pytest.mark.asyncio
async def test_validate_existing_collection_accepts_section_metadata(monkeypatch):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.get.return_value = {"metadatas": [{"chunk_section": "target"}]}
    monkeypatch.setattr(
        db_module,
        "get_existing_collection",
        AsyncMock(return_value=mock_collection),
    )

    await validate_existing_collection(
        "welfare_services_section_aware",
        require_section_metadata=True,
    )

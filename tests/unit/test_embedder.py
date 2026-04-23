from __future__ import annotations

import pytest

from src.embedding.protocol import EmbedderProtocol
from src.embedding.kosimcse import KoSimCSEEmbedder


class MockEmbedder:
    """EmbedderProtocol을 만족하는 테스트용 구현체."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_mock_embedder_satisfies_protocol() -> None:
    mock = MockEmbedder()
    assert isinstance(mock, EmbedderProtocol)


def test_kosimcse_satisfies_protocol() -> None:
    # 실제 모델 로드 없이 Protocol 구조적 서브타이핑만 검증
    assert issubclass(KoSimCSEEmbedder, EmbedderProtocol)


def test_mock_embed_returns_correct_shape() -> None:
    mock = MockEmbedder()
    texts = ["안녕하세요", "복지 서비스"]
    result = mock.embed(texts)

    assert isinstance(result, list)
    assert len(result) == 2
    for vec in result:
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)


def test_mock_embed_empty_input() -> None:
    mock = MockEmbedder()
    result = mock.embed([])
    assert result == []


def test_protocol_isinstance_check() -> None:
    mock = MockEmbedder()
    # runtime_checkable 덕분에 isinstance 검사 가능
    assert isinstance(mock, EmbedderProtocol)

    # embed 메서드 없는 객체는 False
    class NotAnEmbedder:
        pass

    assert not isinstance(NotAnEmbedder(), EmbedderProtocol)

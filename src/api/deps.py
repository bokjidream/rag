from __future__ import annotations

from fastapi import Request

from src.embedding.protocol import EmbedderProtocol


def get_embedder(request: Request) -> EmbedderProtocol:
    return request.app.state.embedder  # type: ignore[no-any-return]

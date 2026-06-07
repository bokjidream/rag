from __future__ import annotations

from fastapi import Request

from src.api.config import ApiConfig
from src.embedding.protocol import EmbedderProtocol


def get_api_config(request: Request) -> ApiConfig:
    config = getattr(request.app.state, "api_config", None)
    if not isinstance(config, ApiConfig):
        raise RuntimeError("API config has not been initialized")
    return config


def get_embedder(request: Request) -> EmbedderProtocol:
    return request.app.state.embedder  # type: ignore[no-any-return]

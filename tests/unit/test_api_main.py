from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.api.config import ApiConfig
from src.api.main import lifespan


@pytest.mark.asyncio
async def test_lifespan_resolves_config_before_chroma_validation_and_embedder() -> None:
    config = ApiConfig(
        collection_name="welfare_services_section_aware",
        adaptive_fetch=True,
    )
    test_app = SimpleNamespace(state=SimpleNamespace())
    events: list[str] = []

    async def _validate_collection(
        collection_name: str,
        *,
        require_section_metadata: bool,
    ) -> None:
        assert test_app.state.api_config is config
        assert collection_name == config.collection_name
        assert require_section_metadata is True
        events.append("validate")

    def _resolve_config() -> ApiConfig:
        events.append("resolve")
        return config

    def _create_embedder() -> object:
        events.append("embedder")
        return object()

    with (
        patch("src.api.main.resolve_api_config", side_effect=_resolve_config),
        patch(
            "src.api.main.validate_existing_collection",
            new=AsyncMock(side_effect=_validate_collection),
        ),
        patch("src.api.main.KoSimCSEEmbedder", side_effect=_create_embedder),
    ):
        async with lifespan(test_app):  # type: ignore[arg-type]
            assert test_app.state.api_config is config
            assert test_app.state.embedder is not None
            events.append("inside")

    assert events == ["resolve", "validate", "embedder", "inside"]

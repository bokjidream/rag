from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from src.db.chroma import WELFARE_COLLECTION


class ApiConfigError(ValueError):
    """Raised when API runtime configuration is invalid."""


@dataclass(frozen=True)
class ApiConfig:
    collection_name: str
    adaptive_fetch: bool

    @property
    def requires_section_metadata(self) -> bool:
        return self.collection_name != WELFARE_COLLECTION


def resolve_api_config(env: Mapping[str, str] | None = None) -> ApiConfig:
    """Resolve API config once before Chroma startup validation runs."""
    values = os.environ if env is None else env

    collection_name = values.get("WELFARE_COLLECTION_NAME", WELFARE_COLLECTION).strip()
    if not collection_name:
        raise ApiConfigError("WELFARE_COLLECTION_NAME must not be empty")

    adaptive_fetch = collection_name != WELFARE_COLLECTION
    adaptive_override = values.get("WELFARE_ADAPTIVE_FETCH")
    if adaptive_override is not None:
        normalized = adaptive_override.strip().lower()
        if normalized not in {"true", "false"}:
            raise ApiConfigError(
                "WELFARE_ADAPTIVE_FETCH must be 'true' or 'false' when set"
            )
        adaptive_fetch = normalized == "true"

    return ApiConfig(collection_name=collection_name, adaptive_fetch=adaptive_fetch)

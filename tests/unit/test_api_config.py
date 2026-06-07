from __future__ import annotations

import pytest

from src.api.config import ApiConfigError, resolve_api_config
from src.db.chroma import WELFARE_COLLECTION


def test_resolve_api_config_defaults_to_baseline_without_adaptive_fetch() -> None:
    config = resolve_api_config({})

    assert config.collection_name == WELFARE_COLLECTION
    assert config.adaptive_fetch is False
    assert config.requires_section_metadata is False


def test_resolve_api_config_enables_adaptive_fetch_for_non_baseline_collection() -> None:
    config = resolve_api_config({"WELFARE_COLLECTION_NAME": "welfare_services_section_aware"})

    assert config.collection_name == "welfare_services_section_aware"
    assert config.adaptive_fetch is True
    assert config.requires_section_metadata is True


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("false", False),
        (" False ", False),
    ],
)
def test_resolve_api_config_accepts_true_false_adaptive_fetch_override(
    override: str,
    expected: bool,
) -> None:
    config = resolve_api_config(
        {
            "WELFARE_COLLECTION_NAME": "welfare_services_section_aware",
            "WELFARE_ADAPTIVE_FETCH": override,
        }
    )

    assert config.adaptive_fetch is expected


def test_resolve_api_config_rejects_invalid_adaptive_fetch_override() -> None:
    with pytest.raises(ApiConfigError, match="WELFARE_ADAPTIVE_FETCH"):
        resolve_api_config({"WELFARE_ADAPTIVE_FETCH": "yes"})


def test_resolve_api_config_rejects_empty_collection_name() -> None:
    with pytest.raises(ApiConfigError, match="WELFARE_COLLECTION_NAME"):
        resolve_api_config({"WELFARE_COLLECTION_NAME": " "})

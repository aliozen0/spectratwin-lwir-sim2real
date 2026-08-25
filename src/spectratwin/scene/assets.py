"""Minimal placement-facing asset registry (SPEC-003 input).

Scope note: SPEC-003 lists "asset registry" as an input, but the full
license-audited mesh/texture registry is Roadmap Session 6 work and does not
exist yet. This module defines only what deterministic scene *placement*
needs — an asset's project category and footprint size for collision/region
checks — plus a license identifier so an unlicensed asset can never enter a
scene. It intentionally carries no mesh/texture/render binding; that belongs
to the render adapter layer (docs/ARCHITECTURE.md "Rendering"), not here.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

from spectratwin.real_data.taxonomy import PROJECT_CATEGORIES

ASSET_REGISTRY_SCHEMA_VERSION = "spectratwin-scene-asset-registry-v1"


class AssetDescriptor(BaseModel):
    """One placeable asset's category, footprint and license identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(min_length=1)
    category: str
    footprint_length_m: float = Field(gt=0)
    footprint_width_m: float = Field(gt=0)
    license_id: str = Field(min_length=1)

    @field_validator("category")
    @classmethod
    def _category_is_known(cls, value: str) -> str:
        if value not in PROJECT_CATEGORIES:
            raise ValueError(f"category must be one of {PROJECT_CATEGORIES}, got {value!r}")
        return value


class AssetRegistry(BaseModel):
    """Frozen, fingerprinted collection of placeable assets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = ASSET_REGISTRY_SCHEMA_VERSION
    assets: tuple[AssetDescriptor, ...]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("assets")
    @classmethod
    def _asset_ids_are_unique(
        cls, value: tuple[AssetDescriptor, ...]
    ) -> tuple[AssetDescriptor, ...]:
        ids = [asset.asset_id for asset in value]
        if len(ids) != len(set(ids)):
            raise ValueError("asset_id values must be unique")
        return value

    def by_category(self, category: str) -> tuple[AssetDescriptor, ...]:
        return tuple(asset for asset in self.assets if asset.category == category)


def _registry_fingerprint(assets: tuple[AssetDescriptor, ...]) -> str:
    canonical = sorted(
        (
            {
                "asset_id": asset.asset_id,
                "category": asset.category,
                "footprint_length_m": asset.footprint_length_m,
                "footprint_width_m": asset.footprint_width_m,
                "license_id": asset.license_id,
            }
            for asset in assets
        ),
        key=lambda asset: json.dumps(asset, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )
    digest_input = json.dumps(
        {"schema_version": ASSET_REGISTRY_SCHEMA_VERSION, "assets": canonical},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def build_asset_registry(assets: list[AssetDescriptor]) -> AssetRegistry:
    frozen_assets = tuple(assets)
    return AssetRegistry(
        assets=frozen_assets,
        fingerprint=_registry_fingerprint(frozen_assets),
    )

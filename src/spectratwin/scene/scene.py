"""Top-level deterministic scene sampling and structural validation (SPEC-003)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from spectratwin.scene.assets import AssetRegistry
from spectratwin.scene.config import SceneConfig
from spectratwin.scene.geometry import build_four_way_intersection, rects_overlap
from spectratwin.scene.placement import CATEGORY_REGION, PlacedObject, PlacementIssue, place_objects

SCENE_DESCRIPTION_SCHEMA_VERSION = "spectratwin-scene-description-v1"


def _config_fingerprint(config: SceneConfig) -> str:
    digest_input = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


class SceneDescription(BaseModel):
    """A fully sampled, inspectable scene, independent of any renderer.

    Reproducible from ``config`` + ``asset_registry`` + ``sample_seed``
    alone (SPEC-003: "Scene sampling MUST be deterministic from sample
    seed/config/assets").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCENE_DESCRIPTION_SCHEMA_VERSION
    sample_seed: int = Field(ge=0)
    config_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_registry_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    road_layout: str
    road_arm_length_m: float = Field(gt=0)
    road_width_m: float = Field(gt=0)
    sidewalk_width_m: float = Field(gt=0)
    min_clearance_m: float = Field(ge=0)
    objects: tuple[PlacedObject, ...]
    placement_issues: tuple[PlacementIssue, ...]


def sample_scene(
    *, config: SceneConfig, asset_registry: AssetRegistry, sample_seed: int
) -> SceneDescription:
    """Deterministically sample one scene from ``config``/``asset_registry``/``sample_seed``.

    ``sample_seed`` is a single resolved integer (SPEC-003 Inputs: "sample
    seed") - callers deriving it from a project master seed use
    :func:`spectratwin.randomness.seed.derive_subseed` upstream, one call
    per ``sample_id``, so this function stays a pure deterministic map from
    seed to scene.
    """
    if sample_seed < 0:
        raise ValueError("sample_seed must be non-negative")

    rng = np.random.default_rng(sample_seed)
    road_network = build_four_way_intersection(config)
    objects, issues = place_objects(
        config=config, asset_registry=asset_registry, road_network=road_network, rng=rng
    )

    return SceneDescription(
        sample_seed=sample_seed,
        config_fingerprint=_config_fingerprint(config),
        asset_registry_fingerprint=asset_registry.fingerprint,
        road_layout=config.road_layout,
        road_arm_length_m=config.road_arm_length_m,
        road_width_m=config.road_width_m,
        sidewalk_width_m=config.sidewalk_width_m,
        min_clearance_m=config.min_clearance_m,
        objects=objects,
        placement_issues=issues,
    )


def validate_scene(scene: SceneDescription, asset_registry: AssetRegistry) -> tuple[str, ...]:
    """Return a tuple of violation descriptions; empty means the scene is valid.

    Re-derives the road network from the scene's own persisted layout
    fields (not from a caller-supplied config) so validation only depends
    on what was actually recorded.
    """
    violations: list[str] = []

    if scene.road_layout != "four_way_intersection":
        return (f"unsupported road_layout {scene.road_layout!r}",)

    road_network = build_four_way_intersection(
        SceneConfig(
            road_layout="four_way_intersection",
            road_arm_length_m=scene.road_arm_length_m,
            road_width_m=scene.road_width_m,
            sidewalk_width_m=scene.sidewalk_width_m,
            object_count_priors={},
            min_clearance_m=scene.min_clearance_m,
            orientation_jitter_rad=0.0,
            placement_retry_budget=1,
        )
    )

    assets_by_id = {asset.asset_id: asset for asset in asset_registry.assets}

    for index, obj in enumerate(scene.objects):
        expected_region = CATEGORY_REGION.get(obj.category)
        if expected_region is None:
            violations.append(f"object[{index}]: unknown category {obj.category!r}")
            continue

        asset = assets_by_id.get(obj.asset_id)
        if asset is None:
            violations.append(f"object[{index}]: asset_id {obj.asset_id!r} not in registry")
        elif asset.category != obj.category:
            violations.append(
                f"object[{index}]: category {obj.category!r} does not match registry "
                f"asset category {asset.category!r}"
            )
        elif (
            asset.footprint_length_m != obj.footprint_length_m
            or asset.footprint_width_m != obj.footprint_width_m
        ):
            violations.append(
                f"object[{index}]: footprint does not match registry asset {obj.asset_id!r}"
            )

        own_rect = obj.footprint_rect(0.0)
        rects = (
            road_network.road_rects if expected_region == "road" else road_network.sidewalk_rects
        )
        if not any(rect.contains_rect(own_rect) for rect in rects):
            violations.append(
                f"object[{index}]: footprint is not fully contained in its {expected_region} region"
            )

        for other_index in range(index + 1, len(scene.objects)):
            other = scene.objects[other_index]
            if rects_overlap(
                obj.footprint_rect(scene.min_clearance_m),
                other.footprint_rect(scene.min_clearance_m),
            ):
                violations.append(f"object[{index}] and object[{other_index}] overlap")

    return tuple(violations)

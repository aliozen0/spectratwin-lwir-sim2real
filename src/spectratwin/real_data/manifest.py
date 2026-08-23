"""Deterministic, content-fingerprinted split manifests.

The manifest stores no licensed source bytes. Its portable fingerprint covers
normalized record content and the SHA-256 digest of each referenced image so a
fresh scan can detect training-relevant source changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from spectratwin.real_data.records import FlirSampleRecord
from spectratwin.real_data.split import REAL_BENCHMARK, SPLIT_POLICY_VERSION
from spectratwin.real_data.taxonomy import CATEGORY_MAPPING_VERSION

MANIFEST_SCHEMA_VERSION = "spectratwin-real-manifest-v2"
MANIFEST_FINGERPRINT_ALGORITHM = "sha256-normalized-records-v1"


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["spectratwin-real-manifest-v2"]
    fingerprint_algorithm: Literal["sha256-normalized-records-v1"]
    role: str
    mapping_version: Literal["flir-taxonomy-v1"]
    split_policy_version: Literal["flir-sequence-split-v1"]
    split_seed: int
    sample_ids: tuple[str, ...]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("sample_ids")
    @classmethod
    def _sample_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)):
            raise ValueError("sample_ids must be sorted")
        if len(value) != len(set(value)):
            raise ValueError("sample_ids must be unique")
        return value


def _canonical_record(record: FlirSampleRecord) -> dict[str, object]:
    annotations = sorted(
        (
            {
                "project_category": annotation.project_category,
                "bbox_xywh": list(annotation.bbox_xywh),
            }
            for annotation in record.annotations
        ),
        key=lambda annotation: json.dumps(
            annotation, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )
    return {
        "source_id": record.source_id,
        "sequence_key": record.sequence_key,
        "relative_image_path": record.relative_image_path,
        "width": record.width,
        "height": record.height,
        "image_sha256": record.image_sha256,
        "annotations": annotations,
    }


def _fingerprint(role: str, split_seed: int, records: list[FlirSampleRecord]) -> str:
    canonical_records = sorted(
        (_canonical_record(record) for record in records),
        key=lambda record: json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )
    digest_input = json.dumps(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "fingerprint_algorithm": MANIFEST_FINGERPRINT_ALGORITHM,
            "role": role,
            "mapping_version": CATEGORY_MAPPING_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "split_seed": split_seed,
            "records": canonical_records,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def build_manifest(role: str, records: list[FlirSampleRecord], master_seed: int) -> DatasetManifest:
    sample_ids = tuple(sorted(r.source_id for r in records))
    return DatasetManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        fingerprint_algorithm=MANIFEST_FINGERPRINT_ALGORITHM,
        role=role,
        mapping_version=CATEGORY_MAPPING_VERSION,
        split_policy_version=SPLIT_POLICY_VERSION,
        split_seed=master_seed,
        sample_ids=sample_ids,
        fingerprint=_fingerprint(role, master_seed, records),
    )


def build_manifests(
    split: dict[str, list[FlirSampleRecord]], master_seed: int
) -> dict[str, DatasetManifest]:
    return {role: build_manifest(role, records, master_seed) for role, records in split.items()}


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Persist a manifest as JSON. Refuses to overwrite a frozen benchmark manifest."""
    if manifest.role == REAL_BENCHMARK and path.exists():
        raise FileExistsError(
            f"{path} already exists; real_benchmark manifests are immutable once written"
        )
    path.write_text(manifest.model_dump_json(indent=2))


def read_manifest(path: Path) -> DatasetManifest:
    """Load only the current content-fingerprinted manifest schema."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must contain a JSON object")

    schema_version = payload.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        if schema_version is None:
            detail = "legacy identity-only manifest without source-content checksums"
        else:
            detail = f"unsupported schema {schema_version!r}"
        raise ValueError(
            f"{detail}; regenerate a {MANIFEST_SCHEMA_VERSION!r} manifest from a fresh scan"
        )
    return DatasetManifest.model_validate(payload)

import hashlib
import json
from pathlib import Path

import pytest

from spectratwin.real_data.manifest import (
    MANIFEST_FINGERPRINT_ALGORITHM,
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    build_manifests,
    read_manifest,
    write_manifest,
)
from spectratwin.real_data.records import FlirAnnotationRecord, FlirSampleRecord
from spectratwin.real_data.split import REAL_BENCHMARK, split_records


def _record(source_id: str, sequence_key: str) -> FlirSampleRecord:
    return FlirSampleRecord(
        source_id=source_id,
        sequence_key=sequence_key,
        relative_image_path=source_id,
        width=100,
        height=100,
        image_sha256=hashlib.sha256(source_id.encode()).hexdigest(),
        annotations=(),
    )


def _fixture_records() -> list[FlirSampleRecord]:
    records = []
    for seq in range(10):
        for frame in range(4):
            records.append(_record(f"seq{seq}_f{frame}.jpg", f"seq{seq}"))
    return records


def test_split_keeps_whole_sequences_together():
    records = _fixture_records()
    split = split_records(records, master_seed=42)

    sequence_to_roles: dict[str, set[str]] = {}
    for role, role_records in split.items():
        for record in role_records:
            sequence_to_roles.setdefault(record.sequence_key, set()).add(role)

    assert all(len(roles) == 1 for roles in sequence_to_roles.values())


def test_split_is_deterministic_for_same_seed():
    records = _fixture_records()
    first = split_records(records, master_seed=7)
    second = split_records(records, master_seed=7)

    assert {r.source_id for r in first["real_train"]} == {r.source_id for r in second["real_train"]}


def test_manifest_fingerprint_stable_for_same_input():
    records = _fixture_records()
    split = split_records(records, master_seed=7)
    manifests_a = build_manifests(split, master_seed=7)
    manifests_b = build_manifests(split, master_seed=7)

    assert manifests_a["real_train"].fingerprint == manifests_b["real_train"].fingerprint


def test_manifest_fingerprint_changes_when_annotation_content_changes():
    original = _record("frame.jpg", "sequence")
    edited = original.model_copy(
        update={
            "annotations": (
                FlirAnnotationRecord(project_category="person", bbox_xywh=(1, 2, 3, 4)),
            )
        }
    )

    original_manifest = build_manifests({"real_train": [original]}, master_seed=7)["real_train"]
    edited_manifest = build_manifests({"real_train": [edited]}, master_seed=7)["real_train"]

    assert original_manifest.fingerprint != edited_manifest.fingerprint


def test_manifest_fingerprint_changes_when_image_content_digest_changes():
    original = _record("frame.jpg", "sequence")
    edited = original.model_copy(update={"image_sha256": "f" * 64})

    original_manifest = build_manifests({"real_train": [original]}, master_seed=7)["real_train"]
    edited_manifest = build_manifests({"real_train": [edited]}, master_seed=7)["real_train"]

    assert original_manifest.fingerprint != edited_manifest.fingerprint


def test_manifest_fingerprint_is_stable_across_record_and_annotation_order():
    first = _record("a.jpg", "sequence").model_copy(
        update={
            "annotations": (
                FlirAnnotationRecord(project_category="person", bbox_xywh=(1, 2, 3, 4)),
                FlirAnnotationRecord(project_category="car", bbox_xywh=(5, 6, 7, 8)),
            )
        }
    )
    reordered = first.model_copy(update={"annotations": tuple(reversed(first.annotations))})
    second = _record("b.jpg", "sequence")

    original = build_manifests({"real_train": [first, second]}, master_seed=7)["real_train"]
    reordered_manifest = build_manifests({"real_train": [second, reordered]}, master_seed=7)[
        "real_train"
    ]

    assert original.fingerprint == reordered_manifest.fingerprint


def test_manifest_persists_explicit_content_fingerprint_schema():
    manifest = build_manifests({"real_train": [_record("frame.jpg", "sequence")]}, 7)["real_train"]

    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION
    assert manifest.fingerprint_algorithm == MANIFEST_FINGERPRINT_ALGORITHM
    serialized = manifest.model_dump_json()
    assert "image_sha256" not in serialized
    assert "annotations" not in serialized


def test_read_manifest_rejects_legacy_identity_only_manifest(tmp_path: Path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "role": "real_train",
                "mapping_version": "legacy",
                "split_policy_version": "legacy",
                "split_seed": 7,
                "sample_ids": ["frame.jpg"],
                "fingerprint": "0" * 64,
            }
        )
    )

    with pytest.raises(ValueError, match="legacy identity-only.*fresh scan"):
        read_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fingerprint_algorithm", "unknown"),
        ("mapping_version", "unknown"),
        ("split_policy_version", "unknown"),
    ),
)
def test_read_manifest_rejects_unknown_identity_version(tmp_path: Path, field: str, value: str):
    manifest = build_manifests({"real_train": [_record("frame.jpg", "sequence")]}, 7)["real_train"]
    payload = manifest.model_dump()
    payload[field] = value
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=field):
        read_manifest(path)


def test_dataset_manifest_requires_explicit_schema_version():
    manifest = build_manifests({"real_train": [_record("frame.jpg", "sequence")]}, 7)["real_train"]
    payload = manifest.model_dump()
    payload.pop("schema_version")

    with pytest.raises(ValueError, match="schema_version"):
        DatasetManifest.model_validate(payload)


@pytest.mark.parametrize(
    "sample_ids",
    (("b.jpg", "a.jpg"), ("a.jpg", "a.jpg")),
)
def test_dataset_manifest_requires_canonical_sample_ids(sample_ids: tuple[str, ...]):
    manifest = build_manifests(
        {"real_train": [_record("a.jpg", "sequence"), _record("b.jpg", "sequence")]}, 7
    )["real_train"]
    payload = manifest.model_dump()
    payload["sample_ids"] = sample_ids

    with pytest.raises(ValueError, match="sample_ids must be (sorted|unique)"):
        DatasetManifest.model_validate(payload)


def test_benchmark_manifest_is_immutable_once_written(tmp_path: Path):
    records = _fixture_records()
    split = split_records(records, master_seed=7)
    manifests = build_manifests(split, master_seed=7)
    path = tmp_path / "real_benchmark.json"

    write_manifest(manifests[REAL_BENCHMARK], path)

    with pytest.raises(FileExistsError):
        write_manifest(manifests[REAL_BENCHMARK], path)

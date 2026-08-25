import json
from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from spectratwin.annotation.coco import (
    ANNOTATION_ID_STRIDE,
    AnnotationIdError,
    CategorySchemaMismatchError,
    CocoIdConflictError,
    CocoValidationError,
    build_coco_document,
    build_frame_annotations,
    deterministic_annotation_id,
    project_category_records,
    write_coco_document,
)
from spectratwin.annotation.policy import AnnotationPolicy, ExclusionReason
from spectratwin.real_data.taxonomy import CATEGORY_MAPPING_VERSION


def test_category_records_come_only_from_the_project_taxonomy() -> None:
    assert [category.model_dump(mode="json") for category in project_category_records()] == [
        {"id": 0, "name": "person", "supercategory": "object"},
        {"id": 1, "name": "car", "supercategory": "object"},
        {"id": 2, "name": "bicycle", "supercategory": "object"},
    ]


def test_annotation_id_is_a_pure_sample_and_instance_function() -> None:
    assert deterministic_annotation_id(image_id=12, instance_index=34) == 120_034
    assert deterministic_annotation_id(image_id=12, instance_index=34) == 120_034


@pytest.mark.parametrize(
    ("image_id", "instance_index"),
    [(-1, 0), (0, -1), (0, ANNOTATION_ID_STRIDE)],
)
def test_annotation_id_rejects_values_outside_its_contract(
    image_id: int, instance_index: int
) -> None:
    with pytest.raises(AnnotationIdError):
        deterministic_annotation_id(image_id=image_id, instance_index=instance_index)


def test_frame_records_are_visible_mask_derived_and_explain_occluded_objects() -> None:
    instance_map = np.array(
        [
            [0, 1, 1],
            [2, 2, 0],
        ],
        dtype=np.uint8,
    )
    policy = AnnotationPolicy(min_visible_area_px=2, min_bbox_side_px=1)

    frame = build_frame_annotations(
        sample_index=7,
        file_name="images/000007.png",
        instance_map=instance_map,
        instance_categories={0: "person", 1: "car", 2: "bicycle"},
        policy=policy,
    )

    assert frame.image.id == 7
    assert frame.image.width == 3
    assert frame.image.height == 2
    assert frame.image.annotation_policy == policy
    assert frame.semantic_map.tolist() == [[-1, 0, 0], [1, 1, -1]]
    assert [annotation.id for annotation in frame.annotations] == [70_000, 70_001]
    assert [annotation.area for annotation in frame.annotations] == [2, 2]
    assert [annotation.bbox for annotation in frame.annotations] == [(1, 0, 2, 1), (0, 1, 2, 1)]
    assert frame.annotations[0].segmentation.size == (2, 3)
    assert frame.image.objects[2].visible_area_px == 0
    assert frame.image.objects[2].bbox is None
    assert frame.image.objects[2].included is False
    assert frame.image.objects[2].exclusion_reason is ExclusionReason.ZERO_VISIBLE_AREA


def test_policy_exclusion_keeps_object_metadata_but_not_annotation() -> None:
    frame = build_frame_annotations(
        sample_index=0,
        file_name="image.png",
        instance_map=np.array([[1]], dtype=np.uint8),
        instance_categories={0: "person"},
        policy=AnnotationPolicy(min_visible_area_px=2, min_bbox_side_px=1),
    )

    assert frame.annotations == ()
    assert frame.image.objects[0].exclusion_reason is ExclusionReason.BELOW_VISIBLE_AREA


def test_disconnected_instance_produces_one_annotation() -> None:
    frame = build_frame_annotations(
        sample_index=1,
        file_name="image.png",
        instance_map=np.array([[1, 0, 1]], dtype=np.uint8),
        instance_categories={0: "person"},
        policy=AnnotationPolicy(),
    )

    assert len(frame.annotations) == 1
    assert frame.annotations[0].area == 2
    assert frame.annotations[0].bbox == (0, 0, 3, 1)


def test_document_is_versioned_sorted_and_json_serialisable() -> None:
    later = build_frame_annotations(
        sample_index=9,
        file_name="nine.png",
        instance_map=np.array([[1]], dtype=np.uint8),
        instance_categories={0: "person"},
        policy=AnnotationPolicy(),
    )
    earlier = build_frame_annotations(
        sample_index=2,
        file_name="two.png",
        instance_map=np.array([[1]], dtype=np.uint8),
        instance_categories={0: "person"},
        policy=AnnotationPolicy(),
    )

    document = build_coco_document([later, earlier])
    payload = document.model_dump(mode="json")

    assert document.schema_version == "spectratwin-coco-v1"
    assert document.category_mapping_version == CATEGORY_MAPPING_VERSION
    assert [image["id"] for image in payload["images"]] == [2, 9]
    assert [annotation["id"] for annotation in payload["annotations"]] == [20_000, 90_000]


@pytest.mark.parametrize("file_name", ["/absolute.png", "../escape.png", "a\\b.png", ""])
def test_image_link_must_be_a_portable_relative_path(file_name: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        build_frame_annotations(
            sample_index=0,
            file_name=file_name,
            instance_map=np.zeros((1, 1), dtype=np.uint8),
            instance_categories={},
            policy=AnnotationPolicy(),
        )


def _document(sample_index: int, file_name: str = "image.png"):
    frame = build_frame_annotations(
        sample_index=sample_index,
        file_name=file_name,
        instance_map=np.array([[1]], dtype=np.uint8),
        instance_categories={0: "person"},
        policy=AnnotationPolicy(),
    )
    return build_coco_document([frame])


def test_writer_is_atomic_deterministic_and_idempotent(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    document = _document(1)

    write_coco_document(path, document)
    first_bytes = path.read_bytes()
    write_coco_document(path, document)

    assert path.read_bytes() == first_bytes
    assert json.loads(first_bytes)["annotations"][0]["id"] == 10_000


def test_writer_merges_compatible_records_without_rebasing_ids(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    write_coco_document(path, _document(8, "eight.png"))

    merged = write_coco_document(path, _document(2, "two.png"))

    assert [image.id for image in merged.images] == [2, 8]
    assert [annotation.id for annotation in merged.annotations] == [20_000, 80_000]


def test_writer_rejects_conflicting_deterministic_id_without_changing_file(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    write_coco_document(path, _document(1, "one.png"))
    original = path.read_bytes()

    with pytest.raises(CocoIdConflictError, match="image id 1"):
        write_coco_document(path, _document(1, "different.png"))

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(category_mapping_version="other"),
        lambda doc: doc.update(categories=list(reversed(doc["categories"]))),
        lambda doc: doc["categories"][0].update(name="pedestrian"),
        lambda doc: doc["categories"][0].update(id=99),
        lambda doc: doc["categories"].pop(),
        lambda doc: doc["categories"].append({"id": 3, "name": "bus", "supercategory": "object"}),
    ],
)
def test_writer_rejects_every_existing_category_schema_change(tmp_path, mutate) -> None:
    path = tmp_path / "annotations.json"
    write_coco_document(path, _document(1))
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))
    incompatible_bytes = path.read_bytes()

    with pytest.raises(CategorySchemaMismatchError):
        write_coco_document(path, _document(2))

    assert path.read_bytes() == incompatible_bytes


def test_writer_rejects_malformed_existing_document_without_changing_it(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    malformed = b"{not-json"
    path.write_bytes(malformed)

    with pytest.raises(CocoValidationError):
        write_coco_document(path, _document(1))

    assert path.read_bytes() == malformed


def test_writer_rejects_valid_schema_with_invalid_geometry(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    payload = _document(1).model_dump(mode="json")
    broken = deepcopy(payload)
    broken["annotations"][0]["bbox"] = [0, 0, 0, 1]
    path.write_text(json.dumps(broken))
    original = path.read_bytes()

    with pytest.raises(CocoValidationError):
        write_coco_document(path, _document(2))

    assert path.read_bytes() == original

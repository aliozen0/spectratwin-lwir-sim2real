from copy import deepcopy

import numpy as np
import pytest

from spectratwin.annotation.coco import build_coco_document, build_frame_annotations
from spectratwin.annotation.policy import AnnotationPolicy
from spectratwin.annotation.validator import ValidationCode, validate_coco_document


def _valid_payload() -> dict:
    frame = build_frame_annotations(
        sample_index=2,
        file_name="images/two.png",
        instance_map=np.array([[0, 1], [0, 1]], dtype=np.uint8),
        instance_categories={0: "person", 1: "car"},
        policy=AnnotationPolicy(),
    )
    return build_coco_document([frame]).model_dump(mode="json")


def _codes(payload: object) -> set[ValidationCode]:
    return {issue.code for issue in validate_coco_document(payload)}


def test_validator_accepts_valid_document_and_empty_frame() -> None:
    assert validate_coco_document(_valid_payload()) == ()

    empty = build_frame_annotations(
        sample_index=3,
        file_name="empty.png",
        instance_map=np.zeros((2, 3), dtype=np.uint8),
        instance_categories={},
        policy=AnnotationPolicy(),
    )
    assert validate_coco_document(build_coco_document([empty]).model_dump(mode="json")) == ()


def test_validator_rejects_non_mapping_document() -> None:
    assert ValidationCode.DOCUMENT_SHAPE in _codes([])


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda doc: doc.update(schema_version="old"), ValidationCode.SCHEMA_VERSION),
        (
            lambda doc: doc.update(categories=list(reversed(doc["categories"]))),
            ValidationCode.CATEGORY_SCHEMA,
        ),
        (
            lambda doc: doc["images"].append(deepcopy(doc["images"][0])),
            ValidationCode.DUPLICATE_IMAGE_ID,
        ),
        (
            lambda doc: doc["annotations"].append(deepcopy(doc["annotations"][0])),
            ValidationCode.DUPLICATE_ANNOTATION_ID,
        ),
        (lambda doc: doc["images"][0].update(width=0), ValidationCode.IMAGE_GEOMETRY),
        (lambda doc: doc["images"][0].update(file_name="../escape"), ValidationCode.IMAGE_PATH),
        (
            lambda doc: doc["images"][0]["annotation_policy"].update(min_visible_area_px=0),
            ValidationCode.POLICY,
        ),
        (
            lambda doc: doc["images"][0]["objects"][0].update(instance_id=99),
            ValidationCode.OBJECT_METADATA,
        ),
        (
            lambda doc: doc["annotations"][0].update(image_id=999),
            ValidationCode.ANNOTATION_REFERENCE,
        ),
        (
            lambda doc: doc["annotations"][0].update(category_id=999),
            ValidationCode.CATEGORY_REFERENCE,
        ),
        (lambda doc: doc["annotations"][0].update(id=123), ValidationCode.ANNOTATION_ID),
        (
            lambda doc: doc["annotations"][0].update(bbox=[0, 0, 0, 1]),
            ValidationCode.BBOX_GEOMETRY,
        ),
        (lambda doc: doc["annotations"][0].update(area=0), ValidationCode.AREA),
        (
            lambda doc: doc["annotations"][0]["segmentation"].update(counts=[1]),
            ValidationCode.SEGMENTATION,
        ),
        (
            lambda doc: doc["annotations"][0].update(bbox=[0, 0, 2, 2]),
            ValidationCode.MASK_GEOMETRY,
        ),
        (
            lambda doc: doc["images"][0]["objects"][0].update(
                included=False, exclusion_reason="below_visible_area"
            ),
            ValidationCode.OBJECT_METADATA,
        ),
    ],
)
def test_validator_reports_each_stable_violation_class(mutate, expected: ValidationCode) -> None:
    payload = _valid_payload()
    mutate(payload)

    assert expected in _codes(payload)


def test_validator_accumulates_independent_issues() -> None:
    payload = _valid_payload()
    payload["schema_version"] = "old"
    payload["images"][0]["width"] = 0

    codes = _codes(payload)

    assert ValidationCode.SCHEMA_VERSION in codes
    assert ValidationCode.IMAGE_GEOMETRY in codes

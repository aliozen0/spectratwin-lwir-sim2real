import numpy as np
import pytest
from pydantic import ValidationError

from spectratwin.annotation.masks import (
    AnnotationMaskError,
    UnknownCategoryError,
    UnmappedInstanceError,
    derive_semantic_map,
    extract_visible_instances,
)
from spectratwin.annotation.policy import AnnotationPolicy, ExclusionReason


def test_split_instance_has_one_tight_visible_bbox() -> None:
    instance_map = np.array(
        [
            [0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )

    visible = extract_visible_instances(instance_map)

    assert len(visible) == 1
    assert visible[0].instance_index == 0
    assert visible[0].instance_id == 1
    assert visible[0].visible_area_px == 3
    assert visible[0].bbox_xywh == (1, 0, 4, 2)
    assert visible[0].truncated is True


def test_single_pixel_and_empty_frame_are_valid() -> None:
    one_pixel = np.zeros((3, 4), dtype=np.uint16)
    one_pixel[1, 2] = 3

    assert extract_visible_instances(one_pixel)[0].bbox_xywh == (2, 1, 1, 1)
    assert extract_visible_instances(one_pixel)[0].truncated is False
    assert extract_visible_instances(np.zeros((2, 3), dtype=np.uint8)) == ()


def test_border_contact_checks_every_image_edge() -> None:
    for row, column in ((0, 2), (3, 2), (2, 0), (2, 4)):
        instance_map = np.zeros((4, 5), dtype=np.uint8)
        instance_map[row, column] = 1
        assert extract_visible_instances(instance_map)[0].truncated is True


@pytest.mark.parametrize(
    "instance_map",
    [
        np.zeros((2, 2, 1), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=bool),
        np.array([[0, -1]], dtype=np.int16),
    ],
)
def test_invalid_instance_maps_are_rejected(instance_map: np.ndarray) -> None:
    with pytest.raises(AnnotationMaskError):
        extract_visible_instances(instance_map)


def test_semantic_map_uses_project_ids_and_minus_one_background() -> None:
    instance_map = np.array([[0, 1, 2, 3]], dtype=np.uint8)

    semantic = derive_semantic_map(
        instance_map,
        {0: "person", 1: "car", 2: "bicycle"},
    )

    assert semantic.dtype == np.int16
    assert semantic.tolist() == [[-1, 0, 1, 2]]


def test_visible_instance_without_mapping_is_rejected() -> None:
    with pytest.raises(UnmappedInstanceError, match="instance_id 2"):
        derive_semantic_map(np.array([[0, 2]], dtype=np.uint8), {0: "person"})


def test_unknown_category_is_rejected_even_when_fully_occluded() -> None:
    with pytest.raises(UnknownCategoryError, match="bus"):
        derive_semantic_map(np.zeros((2, 2), dtype=np.uint8), {0: "bus"})


def test_policy_thresholds_are_inclusive_and_schema_is_persisted() -> None:
    policy = AnnotationPolicy(min_visible_area_px=4, min_bbox_side_px=2)

    assert policy.exclusion_reason(visible_area_px=4, bbox_xywh=(1, 1, 2, 2)) is None
    assert policy.schema_version == "spectratwin-annotation-policy-v1"
    assert AnnotationPolicy(**policy.model_dump(mode="json")) == policy


def test_policy_has_stable_exclusion_reasons() -> None:
    policy = AnnotationPolicy(min_visible_area_px=4, min_bbox_side_px=2)

    assert (
        policy.exclusion_reason(visible_area_px=0, bbox_xywh=None)
        is ExclusionReason.ZERO_VISIBLE_AREA
    )
    assert (
        policy.exclusion_reason(visible_area_px=3, bbox_xywh=(0, 0, 3, 1))
        is ExclusionReason.BELOW_VISIBLE_AREA
    )
    assert (
        policy.exclusion_reason(visible_area_px=4, bbox_xywh=(0, 0, 4, 1))
        is ExclusionReason.BELOW_BBOX_SIDE
    )


@pytest.mark.parametrize(
    ("area", "side"),
    [(0, 1), (1, 0)],
)
def test_policy_rejects_non_positive_thresholds(area: int, side: int) -> None:
    with pytest.raises(ValidationError):
        AnnotationPolicy(min_visible_area_px=area, min_bbox_side_px=side)

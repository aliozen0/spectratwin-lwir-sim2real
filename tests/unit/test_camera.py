import math

import numpy as np
import pytest
from pydantic import ValidationError

from spectratwin.camera.intrinsics import (
    build_intrinsics,
    focal_px_to_fov_deg,
    hfov_deg_to_focal_px,
)
from spectratwin.camera.pose import BoundedRange, CameraPose, CameraPoseConfig, sample_camera_pose
from spectratwin.camera.transform import (
    camera_to_world,
    project_to_pixel,
    project_world_point,
    rotation_matrix,
    world_to_camera,
)

# --- Intrinsics ---------------------------------------------------------------


def test_hfov_to_focal_matches_pinhole_formula():
    focal_px = hfov_deg_to_focal_px(640, 45.0)
    expected = (640 / 2.0) / math.tan(math.radians(45.0) / 2.0)
    assert focal_px == pytest.approx(expected)


def test_hfov_focal_round_trip():
    focal_px = hfov_deg_to_focal_px(640, 45.0)
    recovered_hfov = focal_px_to_fov_deg(640, focal_px)
    assert recovered_hfov == pytest.approx(45.0)


def test_build_intrinsics_default_principal_point_is_image_center():
    intrinsics = build_intrinsics(width_px=640, height_px=512, hfov_deg=45.0)
    assert intrinsics.principal_point_px == (320.0, 256.0)


def test_build_intrinsics_matrix_is_square_pixel_pinhole():
    intrinsics = build_intrinsics(width_px=640, height_px=512, hfov_deg=45.0)
    k = intrinsics.matrix()
    assert k.shape == (3, 3)
    assert k[0, 0] == pytest.approx(intrinsics.focal_length_px)
    assert k[1, 1] == pytest.approx(intrinsics.focal_length_px)
    assert k[0, 2] == pytest.approx(320.0)
    assert k[1, 2] == pytest.approx(256.0)
    assert k[2, 2] == pytest.approx(1.0)
    assert k[0, 1] == 0.0


def test_build_intrinsics_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="width_px"):
        build_intrinsics(width_px=0, height_px=512, hfov_deg=45.0)


def test_hfov_deg_to_focal_px_rejects_out_of_range_fov():
    with pytest.raises(ValueError, match="hfov_deg"):
        hfov_deg_to_focal_px(640, 180.0)
    with pytest.raises(ValueError, match="hfov_deg"):
        hfov_deg_to_focal_px(640, 0.0)


def test_intrinsics_rejects_principal_point_outside_image():
    with pytest.raises(ValidationError):
        build_intrinsics(
            width_px=640, height_px=512, hfov_deg=45.0, principal_point_px=(700.0, 256.0)
        )


# --- Pose ------------------------------------------------------------------------


def _pose_config(**overrides):
    defaults = {
        "location_x_range_m": BoundedRange(min_value=-10.0, max_value=10.0),
        "location_y_range_m": BoundedRange(min_value=-30.0, max_value=-10.0),
        "height_range_m": BoundedRange(min_value=2.5, max_value=4.0),
        "pitch_deg_range": BoundedRange(min_value=5.0, max_value=20.0),
        "yaw_deg_range": BoundedRange(min_value=-15.0, max_value=15.0),
    }
    defaults.update(overrides)
    return CameraPoseConfig(**defaults)


def test_sample_camera_pose_is_deterministic_for_same_seed():
    config = _pose_config()
    pose_a = sample_camera_pose(config, sample_seed=3)
    pose_b = sample_camera_pose(config, sample_seed=3)
    assert pose_a == pose_b


def test_sample_camera_pose_differs_across_seeds():
    config = _pose_config()
    poses = {sample_camera_pose(config, sample_seed=s).model_dump_json() for s in range(5)}
    assert len(poses) == 5


def test_sample_camera_pose_respects_bounds():
    config = _pose_config()
    for seed in range(50):
        pose = sample_camera_pose(config, sample_seed=seed)
        x, y, z = pose.position_m
        assert config.location_x_range_m.min_value <= x <= config.location_x_range_m.max_value
        assert config.location_y_range_m.min_value <= y <= config.location_y_range_m.max_value
        assert config.height_range_m.min_value <= z <= config.height_range_m.max_value
        assert (
            config.pitch_deg_range.min_value <= pose.pitch_deg <= config.pitch_deg_range.max_value
        )
        assert config.yaw_deg_range.min_value <= pose.yaw_deg <= config.yaw_deg_range.max_value


def test_sample_camera_pose_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        sample_camera_pose(_pose_config(), sample_seed=-1)


def test_bounded_range_rejects_inverted_bounds():
    with pytest.raises(ValidationError, match="max_value"):
        BoundedRange(min_value=5.0, max_value=1.0)


def test_camera_pose_config_rejects_non_positive_height():
    with pytest.raises(ValidationError, match="height_range_m"):
        _pose_config(height_range_m=BoundedRange(min_value=-1.0, max_value=2.0))


def test_camera_pose_config_rejects_out_of_range_angles():
    with pytest.raises(ValidationError, match="within"):
        _pose_config(yaw_deg_range=BoundedRange(min_value=-200.0, max_value=10.0))


# --- Transform ---------------------------------------------------------------


def test_rotation_matrix_is_orthonormal_for_arbitrary_pose():
    pose = CameraPose(sample_seed=0, position_m=(1.0, 2.0, 3.0), pitch_deg=12.0, yaw_deg=-37.0)
    r = rotation_matrix(pose)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-9)


def test_yaw_and_pitch_sign_conventions_match_documented_directions():
    """Pins the sign conventions transform.py's module docstring states."""
    # yaw=+90 must look along world -x (west), not +x.
    yawed = CameraPose(sample_seed=0, position_m=(0.0, 0.0, 0.0), pitch_deg=0.0, yaw_deg=90.0)
    forward = rotation_matrix(yawed)[:, 2]
    assert np.allclose(forward, [-1.0, 0.0, 0.0], atol=1e-9)

    # yaw=0 must look along world +y (north).
    level = CameraPose(sample_seed=0, position_m=(0.0, 0.0, 0.0), pitch_deg=0.0, yaw_deg=0.0)
    assert np.allclose(rotation_matrix(level)[:, 2], [0.0, 1.0, 0.0], atol=1e-9)

    # Positive pitch must tilt the optical axis downward (negative z).
    pitched = CameraPose(sample_seed=0, position_m=(0.0, 0.0, 0.0), pitch_deg=20.0, yaw_deg=0.0)
    assert rotation_matrix(pitched)[2, 2] < 0.0


def test_world_to_camera_camera_to_world_round_trip():
    poses = [
        CameraPose(sample_seed=0, position_m=(0.0, 0.0, 3.0), pitch_deg=0.0, yaw_deg=0.0),
        CameraPose(sample_seed=1, position_m=(2.0, -5.0, 3.5), pitch_deg=15.0, yaw_deg=40.0),
        CameraPose(sample_seed=2, position_m=(-4.0, 10.0, 2.5), pitch_deg=-10.0, yaw_deg=-90.0),
    ]
    points = [(0.0, 0.0, 0.0), (5.0, 5.0, 0.0), (-3.0, 20.0, 1.7)]

    for pose in poses:
        for point in points:
            camera_point = world_to_camera(pose, point)
            recovered = camera_to_world(pose, camera_point)
            assert np.allclose(recovered, point, atol=1e-9)


def test_point_directly_ahead_at_camera_height_projects_to_principal_point():
    intrinsics = build_intrinsics(width_px=640, height_px=512, hfov_deg=45.0)
    pose = CameraPose(sample_seed=0, position_m=(0.0, 0.0, 3.0), pitch_deg=0.0, yaw_deg=0.0)

    pixel = project_world_point(intrinsics, pose, (0.0, 25.0, 3.0))

    assert pixel is not None
    assert pixel == pytest.approx(intrinsics.principal_point_px, abs=1e-6)


def test_project_to_pixel_returns_none_when_point_behind_camera():
    intrinsics = build_intrinsics(width_px=640, height_px=512, hfov_deg=45.0)
    behind = np.array([0.0, 0.0, -1.0])
    assert project_to_pixel(intrinsics, behind) is None


def test_increasing_pitch_moves_a_fixed_ahead_point_up_in_frame():
    # Positive pitch tilts the optical axis toward the ground, so a fixed
    # world point ahead must shift monotonically toward smaller v (up).
    intrinsics = build_intrinsics(width_px=640, height_px=512, hfov_deg=45.0)
    point_world = (0.0, 25.0, 0.0)

    v_values = []
    for pitch_deg in (0.0, 5.0, 15.0):
        pose = CameraPose(
            sample_seed=0, position_m=(0.0, 0.0, 3.0), pitch_deg=pitch_deg, yaw_deg=0.0
        )
        pixel = project_world_point(intrinsics, pose, point_world)
        assert pixel is not None
        v_values.append(pixel[1])

    assert v_values[0] > v_values[1] > v_values[2]

    # At pitch=0 the ground point (below camera height) must fall below
    # image center.
    _, cy = intrinsics.principal_point_px
    assert v_values[0] > cy

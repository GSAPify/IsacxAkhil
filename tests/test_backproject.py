import math

import numpy as np
import pytest

from isaacx.perception.backproject import (
    Intrinsics,
    backproject,
    box_depth,
    intrinsics_from_fov,
    look_at,
    project,
    to_world,
)

# Matches configs/default.yaml camera: 1280x720, 60 degree field of view.
K = intrinsics_from_fov(1280, 720, 60.0)


def test_intrinsics_from_fov() -> None:
    assert K.fx == pytest.approx(640.0 / math.tan(math.radians(30.0)))
    assert K.fx == K.fy
    assert (K.cx, K.cy) == (640.0, 360.0)


def test_principal_point_lies_on_optical_axis() -> None:
    np.testing.assert_allclose(backproject(K.cx, K.cy, 2.0, K), [0.0, 0.0, 2.0])


def test_backproject_hand_computed() -> None:
    k = Intrinsics(fx=500.0, fy=400.0, cx=320.0, cy=240.0)
    # (420 - 320) * 1.5 / 500 = 0.3, (140 - 240) * 1.5 / 400 = -0.375
    np.testing.assert_allclose(backproject(420.0, 140.0, 1.5, k), [0.3, -0.375, 1.5])


@pytest.mark.parametrize("point", [[0.2, -0.1, 1.0], [-0.5, 0.3, 2.5], [0.0, 0.0, 0.4]])
def test_project_round_trip(point: list[float]) -> None:
    u, v = project(point, K)
    np.testing.assert_allclose(backproject(u, v, point[2], K), point, atol=1e-12)


def test_non_positive_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="depth"):
        backproject(0.0, 0.0, 0.0, K)
    with pytest.raises(ValueError, match="front"):
        project([0.0, 0.0, -1.0], K)


def test_look_at_centre_pixel_hits_target() -> None:
    position, target = np.array([1.5, 0.0, 1.2]), np.array([0.0, 0.0, 0.5])
    camera_to_world = look_at(position, target)
    distance = float(np.linalg.norm(target - position))
    world = to_world(backproject(K.cx, K.cy, distance, K), camera_to_world)
    np.testing.assert_allclose(world, target, atol=1e-12)


def test_look_at_rotation_is_orthonormal_and_image_up_is_world_up() -> None:
    rotation = look_at([1.5, 0.0, 1.2], [0.0, 0.0, 0.5])[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    # Camera -y is image up, which should point upward in the world.
    assert (rotation @ [0.0, -1.0, 0.0])[2] > 0.0


def test_look_at_straight_down_needs_another_up() -> None:
    with pytest.raises(ValueError, match="parallel"):
        look_at([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
    look_at([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], up=(1.0, 0.0, 0.0))


def test_box_depth_ignores_rim_holes_and_background() -> None:
    depth = np.full((100, 100), 3.0)  # table far behind
    depth[20:80, 20:80] = 1.2  # bin opening
    depth[20:80, 20:24] = 0.9  # near rim on the box edge
    depth[50, 50] = 0.0  # dropout
    depth[51, 51] = np.nan
    assert box_depth(depth, (20.0, 20.0, 80.0, 80.0)) == pytest.approx(1.2)


def test_box_depth_without_valid_pixels_raises() -> None:
    with pytest.raises(ValueError, match="no valid depth"):
        box_depth(np.zeros((10, 10)), (0.0, 0.0, 10.0, 10.0))


def test_bin_position_from_box_and_depth() -> None:
    # A bin centre 1.4 m in front of a camera at the default pose, seen in the image.
    camera_to_world = look_at([1.5, 0.0, 1.2], [0.0, 0.0, 0.5])
    bin_camera = np.array([0.1, 0.05, 1.4])
    u, v = project(bin_camera, K)
    depth = np.full((720, 1280), 4.0)
    depth[int(v) - 30 : int(v) + 30, int(u) - 40 : int(u) + 40] = 1.4
    z = box_depth(depth, (u - 40.0, v - 30.0, u + 40.0, v + 30.0))
    world = to_world(backproject(u, v, z, K), camera_to_world)
    np.testing.assert_allclose(world, to_world(bin_camera, camera_to_world), atol=1e-9)

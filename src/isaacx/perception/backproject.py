"""Pinhole back-projection from pixel + depth to camera and world coordinates.

Camera frame is the OpenCV convention: x right, y down, z forward along the optical axis.
Isaac Sim's USD cameras look down -z with +y up; converting their poses into this convention
belongs to the scene code, not here.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec = NDArray[np.float64]


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> Intrinsics:
    """Square-pixel intrinsics from image size and horizontal field of view."""
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError("hfov_deg must be strictly between 0 and 180")
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Intrinsics(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)


def backproject(u: float, v: float, depth: float, k: Intrinsics) -> Vec:
    """Camera-frame point for pixel (u, v) at `depth` metres along the optical axis."""
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    return np.array([(u - k.cx) * depth / k.fx, (v - k.cy) * depth / k.fy, depth])


def project(point: ArrayLike, k: Intrinsics) -> tuple[float, float]:
    """Pixel (u, v) of a camera-frame point in front of the camera."""
    x, y, z = np.asarray(point, dtype=np.float64)
    if z <= 0.0:
        raise ValueError("point must be in front of the camera (z > 0)")
    return (k.fx * x / z + k.cx, k.fy * y / z + k.cy)


def look_at(position: ArrayLike, target: ArrayLike, up: ArrayLike = (0.0, 0.0, 1.0)) -> Vec:
    """4x4 camera-to-world transform for a camera at `position` looking at `target` (z-up world)."""
    eye = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        raise ValueError("view direction is parallel to up")
    right /= norm
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, down, forward])
    transform[:3, 3] = eye
    return transform


def to_world(point: ArrayLike, camera_to_world: ArrayLike) -> Vec:
    """Apply a 4x4 camera-to-world transform to a camera-frame point."""
    transform = np.asarray(camera_to_world, dtype=np.float64)
    return transform[:3, :3] @ np.asarray(point, dtype=np.float64) + transform[:3, 3]


def box_depth(
    depth: ArrayLike, box: tuple[float, float, float, float], shrink: float = 0.5
) -> float:
    """Median valid depth in the centre of an xyxy box.

    Only the middle `shrink` fraction of the box is used, so the bin's rim and the table behind
    it do not drag the estimate. Zero and non-finite depths are ignored.
    """
    image = np.asarray(depth, dtype=np.float64)
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * (1.0 - shrink) / 2.0, (y1 - y0) * (1.0 - shrink) / 2.0
    rows = slice(max(int(y0 + my), 0), max(int(math.ceil(y1 - my)), 0))
    cols = slice(max(int(x0 + mx), 0), max(int(math.ceil(x1 - mx)), 0))
    patch = image[rows, cols]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise ValueError("no valid depth inside the box")
    return float(np.median(valid))

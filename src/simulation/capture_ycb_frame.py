# pyright: reportMissingImports=false, reportCallIssue=false
"""Capture a single RGB-D frame and 6-DoF object pose for a YCB asset.

Phase 1 deliverable from ``README.md``: replace primitives with a YCB USD
asset, add a camera, and write one RGB-D frame plus ground-truth 6-DoF
pose JSON to ``data/raw/``.

The first iteration mounts an overhead "wrist-like" camera above the
table rather than reparenting it to ``panda_hand``. Reparenting onto the
Franka end-effector is the next refinement once the capture pipeline is
verified end-to-end.

Outputs (under ``data/raw/<UTC-timestamp>/``):
    - ``rgb.png``        - 8-bit RGB.
    - ``depth.png``      - 16-bit single-channel depth in millimetres.
    - ``metadata.json``  - object pose (wxyz quaternion + xyz translation),
                           camera pose/intrinsics, asset root, capture time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"

CAMERA_RESOLUTION: Tuple[int, int] = (1280, 720)
CAMERA_FOV_DEG = 60.0
CAMERA_POSITION: Tuple[float, float, float] = (0.5, 0.0, 1.5)
CAMERA_LOOKAT: Tuple[float, float, float] = (0.5, 0.0, 0.4)
CAMERA_HORIZONTAL_APERTURE_MM = 24.0
CAMERA_NEAR_M = 0.05
CAMERA_FAR_M = 10.0

TABLE_POSITION: Tuple[float, float, float] = (0.5, 0.0, 0.35)
TABLE_SCALE: Tuple[float, float, float] = (1.0, 0.6, 0.1)
TABLE_TOP_Z = TABLE_POSITION[2] + TABLE_SCALE[2] * 0.5

YCB_OBJECT_ID = "004_sugar_box"
YCB_ASSET_RELATIVE = f"/Isaac/Props/YCB/Axis_Aligned_Physics/{YCB_OBJECT_ID}.usd"
YCB_PRIM_PATH = "/World/YCB/SugarBox"
YCB_DROP_POSITION: Tuple[float, float, float] = (
    TABLE_POSITION[0],
    TABLE_POSITION[1],
    TABLE_TOP_Z + 0.10,
)

SETTLE_STEPS = 120


@dataclass(frozen=True)
class CaptureMetadata:
    object_id: str
    object_prim: str
    object_position_xyz_m: Tuple[float, float, float]
    object_orientation_wxyz: Tuple[float, float, float, float]
    camera_position_xyz_m: Tuple[float, float, float]
    camera_lookat_xyz_m: Tuple[float, float, float]
    camera_resolution_wh_px: Tuple[int, int]
    camera_horizontal_fov_deg: float
    captured_at_utc: str
    asset_root: str


simulation_app: Any = SimulationApp(  # type: ignore[reportCallIssue]
    {
        "headless": True,
        "hide_ui": True,
        "enable_crashreporter": False,
        "multi_gpu": False,
        "physics_gpu": 0,
    }
)

import cv2
from scipy.spatial.transform import Rotation

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path


def main() -> None:
    print("Resolving Isaac Sim assets root...", flush=True)
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not find Isaac Sim assets root path.")

    print("Creating world...", flush=True)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    print("Adding table...", flush=True)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=list(TABLE_POSITION),
            scale=list(TABLE_SCALE),
            color=np.array([0.45, 0.30, 0.18]),
        )
    )

    print(f"Adding YCB asset {YCB_OBJECT_ID}...", flush=True)
    add_reference_to_stage(
        usd_path=assets_root_path + YCB_ASSET_RELATIVE,
        prim_path=YCB_PRIM_PATH,
    )
    ycb = XFormPrim(
        prim_paths_expr=YCB_PRIM_PATH,
        positions=np.array([list(YCB_DROP_POSITION)], dtype=np.float64),
    )

    print("Adding Franka...", flush=True)
    add_reference_to_stage(
        usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd",
        prim_path="/World/Franka",
    )

    print("Adding capture camera...", flush=True)
    camera_orientation_wxyz = _lookat_quaternion(CAMERA_POSITION, CAMERA_LOOKAT)
    camera = Camera(
        prim_path="/World/CaptureCamera",
        position=np.array(CAMERA_POSITION, dtype=np.float64),
        orientation=camera_orientation_wxyz,
        resolution=CAMERA_RESOLUTION,
        frequency=20,
    )

    print("Resetting world...", flush=True)
    world.reset()

    camera.initialize()
    camera.set_focal_length(_focal_length_mm_for_fov(CAMERA_RESOLUTION[0], CAMERA_FOV_DEG))
    camera.set_clipping_range(CAMERA_NEAR_M, CAMERA_FAR_M)
    camera.add_distance_to_image_plane_to_frame()

    print(f"Stepping simulation for {SETTLE_STEPS} frames...", flush=True)
    for _ in range(SETTLE_STEPS):
        world.step(render=True)

    print("Capturing frame...", flush=True)
    frame = camera.get_current_frame()
    rgb = _frame_to_rgb_uint8(frame)
    depth_mm = _frame_to_depth_mm_uint16(frame)

    object_positions, object_orientations = ycb.get_world_poses()
    object_pos_xyz: Tuple[float, float, float] = (
        float(object_positions[0][0]),
        float(object_positions[0][1]),
        float(object_positions[0][2]),
    )
    object_ori_wxyz: Tuple[float, float, float, float] = (
        float(object_orientations[0][0]),
        float(object_orientations[0][1]),
        float(object_orientations[0][2]),
        float(object_orientations[0][3]),
    )

    capture_dir = _make_capture_dir()
    cv2.imwrite(str(capture_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(capture_dir / "depth.png"), depth_mm)

    metadata = CaptureMetadata(
        object_id=YCB_OBJECT_ID,
        object_prim=YCB_PRIM_PATH,
        object_position_xyz_m=object_pos_xyz,
        object_orientation_wxyz=object_ori_wxyz,
        camera_position_xyz_m=CAMERA_POSITION,
        camera_lookat_xyz_m=CAMERA_LOOKAT,
        camera_resolution_wh_px=CAMERA_RESOLUTION,
        camera_horizontal_fov_deg=CAMERA_FOV_DEG,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        asset_root=assets_root_path,
    )
    (capture_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2))

    print(f"Wrote capture to {capture_dir}", flush=True)
    print("Closing Isaac Sim...", flush=True)
    simulation_app.close()


def _make_capture_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = DATA_RAW / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _focal_length_mm_for_fov(width_px: int, fov_deg: float) -> float:
    fov_rad = np.deg2rad(fov_deg)
    return float(CAMERA_HORIZONTAL_APERTURE_MM / (2.0 * np.tan(fov_rad / 2.0)))


def _lookat_quaternion(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
) -> np.ndarray:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation_matrix = np.column_stack([right, up, -forward])
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _frame_to_rgb_uint8(frame: dict) -> np.ndarray:
    rgba = np.asarray(frame["rgba"])
    return np.ascontiguousarray(rgba[..., :3].astype(np.uint8))


def _frame_to_depth_mm_uint16(frame: dict) -> np.ndarray:
    distance_m = np.asarray(frame["distance_to_image_plane"], dtype=np.float32)
    distance_m = np.where(np.isfinite(distance_m), distance_m, 0.0)
    distance_mm = np.clip(distance_m * 1000.0, 0, 65535).astype(np.uint16)
    return distance_mm


if __name__ == "__main__":
    main()

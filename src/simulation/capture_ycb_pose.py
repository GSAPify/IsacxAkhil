# pyright: reportMissingImports=false, reportCallIssue=false
"""Capture the 6-DoF pose of a YCB asset from a settled scene.

Phase 1 partial deliverable from ``README.md``: spawn the YCB
``004_sugar_box`` USD on a table, settle physics, and write its
world-space 6-DoF pose to ``data/raw/<UTC-timestamp>/metadata.json``.

This is the pose-only fallback that uses the minimal headless Kit
experience at ``configs/isaac/hello_world_headless.kit`` (the same
experience proven to work in ``src/robot_control/test.py``). It
deliberately avoids loading the RTX renderer / MDL pipeline because
those crash with a Vulkan access violation on driver ``595.79`` +
Windows 11 25H2; see ``LOG.md`` 2026-05-23.

Once the driver is rolled to a Kit-supported Studio build, the sibling
``src/simulation/capture_ycb_frame.py`` adds RGB + depth on top of the
same scene.
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
HEADLESS_EXPERIENCE = REPO_ROOT / "configs" / "isaac" / "hello_world_headless.kit"
DATA_RAW = REPO_ROOT / "data" / "raw"

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
class PoseMetadata:
    object_id: str
    object_prim: str
    object_position_xyz_m: Tuple[float, float, float]
    object_orientation_wxyz: Tuple[float, float, float, float]
    captured_at_utc: str
    asset_root: str


simulation_app: Any = SimulationApp(  # type: ignore[reportCallIssue]
    {
        "headless": True,
        "hide_ui": True,
        "disable_viewport_updates": True,
        "enable_crashreporter": False,
        "multi_gpu": False,
        "physics_gpu": 0,
    },
    experience=str(HEADLESS_EXPERIENCE),
)

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
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

    print("Resetting world...", flush=True)
    world.reset()

    print(f"Stepping simulation for {SETTLE_STEPS} frames...", flush=True)
    for _ in range(SETTLE_STEPS):
        world.step(render=False)

    print("Reading object pose...", flush=True)
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
    metadata = PoseMetadata(
        object_id=YCB_OBJECT_ID,
        object_prim=YCB_PRIM_PATH,
        object_position_xyz_m=object_pos_xyz,
        object_orientation_wxyz=object_ori_wxyz,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        asset_root=assets_root_path,
    )
    (capture_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2))

    print(f"Wrote pose to {capture_dir}", flush=True)
    print("Closing Isaac Sim...", flush=True)
    simulation_app.close()


def _make_capture_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = DATA_RAW / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np

from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[2]
HEADLESS_EXPERIENCE = REPO_ROOT / "configs" / "isaac" / "hello_world_headless.kit"

simulation_app = SimulationApp(
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
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path


def main() -> None:
    print("Creating world...", flush=True)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Table as a fixed scaled cube.
    print("Adding table and cube...", flush=True)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=[0.5, 0.0, 0.35],
            scale=[1.0, 0.6, 0.1],
            color=np.array([0.45, 0.30, 0.18]),
        )
    )

    # One dynamic cube sitting above the table.
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=[0.5, 0.0, 0.8],
            scale=[0.08, 0.08, 0.08],
            color=np.array([0.1, 0.4, 1.0]),
        )
    )

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not find Isaac Sim assets root path.")

    print("Adding Franka...", flush=True)
    franka_usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
    add_reference_to_stage(
        usd_path=franka_usd_path,
        prim_path="/World/Franka",
    )

    print("Resetting world...", flush=True)
    world.reset()

    print("Stepping simulation...", flush=True)
    for _ in range(300):
        world.step(render=False)

    print("Closing Isaac Sim...", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
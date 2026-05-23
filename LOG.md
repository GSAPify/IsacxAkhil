# Project Log

Append-only daily/session log for the robotics perception project. Keep entries short, factual, and tied to decisions, setup work, experiments, blockers, and measured results.

## 2026-05-18

- Captured the realistic 12-16 week project plan in `README.md`.
- Added the YCB Object and Model Set as the benchmark/data anchor.
- Documented remote readiness checks for `AKHIL-ASUS`.
- Verified the machine sees an NVIDIA GeForce RTX 4070 SUPER with 12 GB VRAM.
- Found environment blockers: active PyTorch install is CPU-only, and `git`/Rust tools are not currently available in the shell PATH.
- Added repo hygiene ignores for generated datasets, model weights, experiment outputs, and local build artifacts.
- Added the project skeleton: `docs/decisions/`, `docs/results/`, `data/`, `src/`, and `notebooks/`.
- Fixed the Isaac Sim venv PyTorch install to GPU-enabled `torch 2.7.0+cu126` and `torchvision 0.22.0+cu126`.
- Verified `torch.cuda.is_available()` is `True` on the NVIDIA GeForce RTX 4070 SUPER.
- Verified `pip check` reports no broken requirements.
- Committed the repository setup locally; GitHub push is blocked until SSH or HTTPS credentials are available on this machine.

## 2026-05-19

- Started Week 1 Isaac Sim onboarding deliverable: standalone scripted scene.
- Confirmed the default Isaac Sim Python experience crashes from SSH/headless startup in UI/hotkey setup.
- Added a minimal headless Kit experience at `configs/isaac/hello_world_headless.kit` to avoid GUI/menu startup.
- Verified `src/robot_control/test.py` creates a `World`, adds a ground plane, table cuboid, dynamic cube, and Franka USD reference.
- Ran `python src\robot_control\test.py`; the script reset the world, stepped simulation headlessly, closed Isaac Sim, and exited with code 0.

## 2026-05-23

- Investigated launching the Isaac Sim 5.1 GUI on `AKHIL-ASUS` at the desk; first run reached `app ready` at 35.9s but no window persisted.
- Two follow-up runs were poisoned by my own bad CLI: `--/app/quitAfter=0` auto-quits after one frame, and concatenating the relaunch command with the `Remove-Item` config-reset onto a single PowerShell line caused Kit to load `user.config.json` as a stage and hard-crash mid-init.
- Cleared `%LOCALAPPDATA%\ov\data\Kit\Isaac-Sim Full\5.1\user.config.json` and the entire `%LOCALAPPDATA%\ov\cache\` to drop a corrupted post-crash material/shader cache.
- Post-cache-wipe relaunch reaches `app ready` at 10.9s, then hard-crashes at ~11.1s during `rtx.scenedb.plugin` and `rtx.multigpumanager.plugin` initialization. No shutdown sequence in the kit log; the process is killed by the renderer.
- Likely cause is a Vulkan/RTX regression on driver `595.79` (Kit log explicitly recommends `537.58 GameReady/Studio` baseline) combined with hybrid-GPU routing on Windows 11 25H2 with the AMD Radeon iGPU enabled alongside the RTX 4070 SUPER. Driver roll and iGPU disable are deferred to a later session.
- Pivoted to headless for the rest of Phase 1 since none of the remaining Phase 1 deliverables require an interactive viewport.
- Added `src/simulation/capture_ycb_frame.py` mirroring the `src/robot_control/test.py` headless-`SimulationApp` pattern; the script swaps the dynamic cube for the YCB `004_sugar_box` USD, adds a 1280x720 / 60 deg overhead capture camera, settles physics, and writes `rgb.png`, `depth.png` (16-bit mm), and `metadata.json` (object 6-DoF pose, camera pose/intrinsics) into `data/raw/<UTC-timestamp>/`.
- Note for next session: this iteration uses a fixed overhead "wrist-like" camera at `(0.5, 0, 1.5)` rather than parenting onto `panda_hand`; switching to an actual wrist-mounted camera is the next refinement once capture is verified end-to-end.

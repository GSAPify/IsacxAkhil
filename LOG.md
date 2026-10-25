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

## 2026-10-25

- Proposed toss-to-bin as a Phase 7 extension (ADR 003): soft YCB balls thrown into bins, release velocity from ballistics plus a learned residual, inspired by TossingBot.
- Wrote the slice plan in `docs/TOSS_TO_BIN.md`. Slices 1-3 (ballistics, back-projection, residual on a toy simulator) need no Isaac Sim; slices 4-6 run on the workstation.

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

# Project Log

Append-only daily/session log for the robotics perception project. Keep entries short, factual, and tied to decisions, setup work, experiments, blockers, and measured results.

## 2026-05-18

- Captured the realistic 12-16 week project plan in `README.md`.
- Added the YCB Object and Model Set as the benchmark/data anchor.
- Documented remote readiness checks for `AKHIL-ASUS`.
- Verified the machine sees an NVIDIA GeForce RTX 4070 SUPER with 12 GB VRAM.
- Found environment blockers: active PyTorch install is CPU-only, and `git`/Rust tools are not currently available in the shell PATH.
- Added repo hygiene ignores for generated datasets, model weights, experiment outputs, and local build artifacts.

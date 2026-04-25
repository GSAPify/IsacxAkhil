# Robotics

Monorepo for simulation-oriented robotics work: **Python** (Isaac Sim ecosystem, YOLO training/export) and **Rust** (fast ONNX inference via [`vision-inference`](rust/crates/vision-inference)).

## Repository layout

| Path | Purpose |
|------|---------|
| [`configs/default.yaml`](configs/default.yaml) | Target settings for sim, camera, detection, and inference paths (consumers to be wired as the stack grows). |
| [`docs/VISION_PIPELINE.md`](docs/VISION_PIPELINE.md) | Python ↔ ONNX ↔ Rust responsibilities and commands. |
| [`scripts/`](scripts/) | Utilities (e.g. YOLO → ONNX export, GPU stress test). |
| [`rust/`](rust/) | Cargo workspace; [`vision-inference`](rust/crates/vision-inference) loads exported ONNX models. |

Large or generated artifacts (`*.onnx`, `*.pt`, local venvs, build outputs) are listed in [`.gitignore`](.gitignore).

## Prerequisites

- **Windows** (paths in docs assume PowerShell).
- **Python 3.11** and a virtual environment (this repo convention: `env_isaacsim`).
- **Isaac Sim** (optional for inference export): install per [NVIDIA Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/) docs, e.g. `pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com` into that venv.
- **Rust** ([rustup](https://rustup.rs/)) for the `vision-inference` crate. ONNX Runtime native deps typically need **Visual Studio Build Tools** with **Desktop development with C++**.

## Quick start

From the repository root:

```powershell
# Python deps (same venv as Isaac Sim if you use one)
.\activate.ps1
python -m pip install -r requirements.txt
```

Export YOLO weights to ONNX (writes under `models/`; ensure that directory exists or adjust `--out`):

```powershell
python scripts\export_yolo_onnx.py --weights yolov8n.pt --out models\yolov8n.onnx
```

Build and run the Rust inference binary:

```powershell
cd rust
cargo build -p vision-inference --release
.\target\release\vision-inference.exe --model ..\models\yolov8n.onnx --image path\to\image.png
```

More detail: [`docs/VISION_PIPELINE.md`](docs/VISION_PIPELINE.md).

## License

See crate and workspace metadata in [`rust/Cargo.toml`](rust/Cargo.toml) (SPDX: `MIT OR Apache-2.0`).

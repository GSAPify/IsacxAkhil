# Robotics

Monorepo for simulation-oriented robotics work: **Python** (Isaac Sim ecosystem, YOLO training/export) and **Rust** (fast ONNX inference via [`vision-inference`](rust/crates/vision-inference)).

## Project goal

Build a portfolio-grade sim-to-real 6-DoF object pose estimation stack around Isaac Sim, YCB objects, synthetic data, pose estimation, tracking, latency measurement, and a small Rust systems component. The realistic working assumption is **10-15 focused hours per week** alongside a day job, so honest delivery is **12-16 weeks**.

The benchmark/data source to keep anchored in this repo is the [YCB Object and Model Set](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/). Use 3-5 physical YCB counterparts for sim-to-real evaluation; good first objects are `003_cracker_box`, `004_sugar_box`, `006_mustard_bottle`, `010_potted_meat_can`, and `035_power_drill`.

## Repository layout

| Path | Purpose |
|------|---------|
| [`LOG.md`](LOG.md) | Append-only daily/session log for setup work, experiments, blockers, and results. |
| [`configs/default.yaml`](configs/default.yaml) | Target settings for sim, camera, detection, and inference paths (consumers to be wired as the stack grows). |
| [`data/`](data/) | Local datasets and captures; large contents are gitignored. |
| [`docs/decisions/`](docs/decisions/) | ADRs for major implementation choices. |
| [`docs/results/`](docs/results/) | Evaluation tables, latency numbers, and result notes. |
| [`docs/VISION_PIPELINE.md`](docs/VISION_PIPELINE.md) | Python ↔ ONNX ↔ Rust responsibilities and commands. |
| [`notebooks/`](notebooks/) | Sanity-check notebooks for data and pose visualization. |
| [`scripts/`](scripts/) | Utilities (e.g. YOLO → ONNX export, GPU stress test). |
| [`src/`](src/) | Future Python package code for simulation, data, training, and tracking. |
| [`rust/`](rust/) | Cargo workspace; [`vision-inference`](rust/crates/vision-inference) loads exported ONNX models. |

Large or generated artifacts (`*.onnx`, `*.pt`, local venvs, build outputs) are listed in [`.gitignore`](.gitignore).

## Prerequisites

- **Windows** (paths in docs assume PowerShell).
- **Python 3.11** and a virtual environment (this repo convention: `env_isaacsim`).
- **Isaac Sim** (optional for inference export): install per [NVIDIA Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/) docs, e.g. `pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com` into that venv.
- **Rust** ([rustup](https://rustup.rs/)) for the `vision-inference` crate. ONNX Runtime native deps typically need **Visual Studio Build Tools** with **Desktop development with C++**.
- **NVIDIA GPU/driver** visible through `nvidia-smi`; RTX 30-series is the minimum target for Isaac Sim, RTX 40-series is more comfortable.
- **Remote workstation access** to `AKHIL-ASUS` through VS Code/Cursor Remote SSH before starting long Isaac Sim or dataset-generation runs.

## Remote readiness checklist

Run these checks from a fresh PowerShell SSH session on `AKHIL-ASUS` before beginning Phase 1:

```powershell
cd G:\Robotics
nvidia-smi
.\activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
.\scripts\gpu_test.ps1 -MonitorOnly
```

If PyTorch reports `False` for CUDA, reinstall the CUDA wheel inside `env_isaacsim`:

```powershell
.\activate.ps1
python -m pip install --upgrade --force-reinstall torch==2.7.0+cu126 torchvision==0.22.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.__version__, torch.version.cuda)"
```

For Rust:

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
rustc --version
cargo --version
cd G:\Robotics\rust
cargo build -p vision-inference --release
```

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

## Delivery plan

### Phase 1: Foundation, weeks 1-2

Install and validate Isaac Sim on `AKHIL-ASUS`, work through the official tutorials, and build a scripted scene with a Franka arm, table, primitives, and rendered output. Then replace primitives with selected YCB USD assets, mount a wrist camera, and capture one RGB-D frame plus ground-truth 6-DoF pose JSON per object.

Do not move past this phase until scripted spawning and pose ground truth are reliable.

### Phase 2: Synthetic data pipeline, weeks 3-4

Use Isaac Sim Replicator for domain randomization: object pose/count, camera jitter, lighting, table/wall textures, and distractors. First target is 1,000 verified samples with RGB, depth, masks, and poses; then scale to 30,000-50,000 samples. Always visualize random samples with pose overlays before training.

### Phase 3: Pose estimation model, weeks 5-8

Train an end-to-end baseline first so the pipeline is measurable. The preferred portfolio path is keypoint heatmaps plus PnP with known 3D geometry; RGB-D fusion is the alternative if depth becomes central. Track ADD-S on a held-out sim test set, then spend a full week on failure analysis for symmetry, occlusion, and lighting extremes.

### Phase 4: Tracking and state estimation, weeks 9-10

Add multi-object tracking with Hungarian assignment over pose distance and an EKF per object using position, orientation, velocity, and angular velocity. Include track birth/death, occlusion gaps, identity-switch handling, and a confidence gate.

### Phase 5: Real-time loop and Rust, weeks 11-12

Export the trained model to ONNX, measure fixed-Hz Python inference loop latency by stage, and target 30 Hz while reporting actual numbers. Scope the Rust work conservatively: move the tracker into a standalone Rust crate first, then consider full ONNX inference in Rust only if time allows.

### Phase 6: Sim-to-real evaluation and docs, weeks 13-14

Photograph the physical YCB objects on a real table, label a small set, run the sim-trained model, quantify the gap, and ablate domain randomization, photometric augmentation, and small real fine-tuning. Final docs should include the problem, architecture, sim-to-real numbers, latency budget, failure modes, and deployment notes.

If the schedule slips, cut in this order: full Rust inference, multi-object tracking, real-world ablations. Do not cut sim-to-real photo evaluation, latency measurement, or failure analysis.

## License

See crate and workspace metadata in [`rust/Cargo.toml`](rust/Cargo.toml) (SPDX: `MIT OR Apache-2.0`).

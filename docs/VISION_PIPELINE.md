# Vision pipeline: Python (sim + ML) and Rust (inference node)

## Split of responsibilities

| Layer | Language | Role |
|-------|----------|------|
| Simulation, sensors, synthetic data | Python | Isaac Sim, camera streams, ground truth, dataset capture |
| Training, export | Python | Ultralytics / PyTorch, export to ONNX (`scripts/export_yolo_onnx.py`) |
| Fast inference, low-latency I/O | Rust | ONNX Runtime via `ort` (`rust/crates/vision-inference`), optional control / IPC later |
| Tracking (prototype) | Python | SORT / ByteTrack; move to Rust only if hard real-time is needed (see ADR 002) |

Data crosses the boundary as **files or sockets**: ONNX model path + RGB frames (numpy in Python,
`image` / shared memory in Rust). Keep `requirements.txt` for the Python side.

## Flow

1. Python: run the sim or load recorded frames, train or fine-tune YOLO, export `models/yolov8n.onnx`.
2. Rust: `vision-inference` loads that ONNX model and runs it on frames.
3. Later: Python publishes frames (e.g. ZMQ, gRPC, shared memory); Rust subscribes and returns detections.

## Paths

- Config: `configs/default.yaml` keys `detection`, `inference` (`onnx_path`, `imgsz`).
- After export: `models/yolov8n.onnx` (gitignored like other `*.onnx`; keep weights local).

## Verify Rust on Windows (PowerShell over SSH)

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
rustc --version
cargo --version
rustup show
```

If `rustc` is not found, install from https://rustup.rs then open a **new** shell. For native
crates (including `ort`), install **Visual Studio Build Tools** with the **Desktop development
with C++** workload.

## Build and run inference

```powershell
cd G:\Robotics\rust
cargo build -p vision-inference --release
.\target\release\vision-inference.exe --model ..\models\yolov8n.onnx --image path\to\frame.png
```

See the README's Quick start for runtime setup and the output format.

## Export ONNX (Python venv)

```powershell
cd G:\Robotics
.\activate.ps1
python scripts\export_yolo_onnx.py --weights yolov8n.pt --out models\yolov8n.onnx
```

Point `configs/default.yaml` key `inference.onnx_path` at that file, or pass `--model` to the
Rust binary.

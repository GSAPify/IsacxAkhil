# Vision pipeline: Python (sim + ML) and Rust (inference node)

## Split of responsibilities

| Layer | Language | Role |
|-------|----------|------|
| Simulation, sensors, synthetic data | Python | Isaac Sim, camera streams, ground truth, dataset capture |
| Training, export | Python | Ultralytics / PyTorch, export to ONNX |
| Fast inference, low-latency I/O | Rust | ONNX Runtime (`ort`), optional control / IPC later |
| Tracking (prototype) | Python | SORT / ByteTrack; move to Rust only if you need hard real-time |

Data crosses the boundary as **files or sockets**: ONNX model path + RGB frames (numpy in Python, `image` / shared memory in Rust).

## Flow

1. Python: run sim or load recorded frames, train or fine-tune YOLO, export `models/yolov8n.onnx`.
2. Rust: `vision-inference` loads that ONNX, runs inference (postprocess for YOLO can be expanded incrementally).
3. Later: Python publishes frames (e.g. ZMQ, gRPC, shared memory); Rust subscribes and returns detections.

## Verify Rust on Windows (run over SSH in PowerShell)

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
rustc --version
cargo --version
rustup show
```

If `rustc` is not found, install from https://rustup.rs then open a **new** shell. For native crates (including `ort`), install **Visual Studio Build Tools** with the **Desktop development with C++** workload.

## Build the Rust inference crate

```powershell
cd G:\Robotics\rust
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
cargo build -p vision-inference --release
```

## Export ONNX from Python (venv with requirements.txt)

```powershell
.\activate.ps1
python scripts\export_yolo_onnx.py --weights yolov8n.pt --out models\yolov8n.onnx
```

Point `configs/default.yaml` key `inference.onnx_path` at that file, or pass `--model` to the Rust binary.

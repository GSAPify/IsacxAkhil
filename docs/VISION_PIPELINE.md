# Vision pipeline: Python + Rust

## Responsibilities

- **Python** — Isaac Sim, cameras, datasets, Ultralytics/YOLO training, export to ONNX (`scripts/export_yolo_onnx.py`). Keep `requirements.txt` for this side.
- **Rust** — Fast inference node using ONNX Runtime (`rust/crates/vision-inference`). Optional later: sockets/IPC to Python.

## Paths

- Config: `configs/default.yaml` keys `detection`, `inference` (onnx path, imgsz).
- After export: `models/yolov8n.onnx` (gitignored like other `*.onnx`; keep weights local).

## Rust toolchain check (PowerShell over SSH)

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
rustc --version
cargo --version
rustup show
```

Install Rust: https://rustup.rs — then **Visual Studio Build Tools** with **Desktop development with C++** for linking native crates (e.g. `ort`).

## Build and run inference

```powershell
cd G:\Robotics\rust
cargo build -p vision-inference --release
..\rust\target\release\vision-inference.exe --model ..\models\yolov8n.onnx --image path\to\frame.png
```

## Export ONNX (Python venv)

```powershell
cd G:\Robotics
.\activate.ps1
python scripts\export_yolo_onnx.py --weights yolov8n.pt --out models\yolov8n.onnx
```

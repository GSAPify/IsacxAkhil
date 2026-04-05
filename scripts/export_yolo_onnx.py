import argparse
from pathlib import Path

"""Export a YOLO weights file to ONNX for the Rust vision-inference node (Python side)."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Ultralytics YOLO to ONNX")
    parser.add_argument("--weights", default="yolov8n.pt", help="Weights path or hub id (e.g. yolov8n.pt)")
    parser.add_argument("--out", default="models/yolov8n.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Square export size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
    )
    src = Path(exported)
    if not src.is_file():
        raise SystemExit(f"Export did not produce a file: {exported}")
    src.replace(out_path)
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()

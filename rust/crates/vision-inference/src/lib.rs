//! YOLOv8 ONNX inference building blocks: Ultralytics-compatible letterbox preprocessing and
//! output decoding with non-maximum suppression.
//!
//! Everything here is pure and runs without ONNX Runtime, so it is unit-tested directly. The
//! `vision-inference` binary wires it to an `ort` session.

pub mod postprocess;
pub mod preprocess;

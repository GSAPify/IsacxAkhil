//! Run an exported YOLOv8 ONNX model on an image and print the detections as JSON.
//!
//! Without `--image`, runs one zero tensor through the model and logs its output shapes, which
//! is a quick check that the model loads and executes.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use serde::Serialize;
use vision_inference::postprocess::{decode_yolov8, nms, Detection};
use vision_inference::preprocess::preprocess;

#[derive(Parser, Debug)]
#[command(
    name = "vision-inference",
    about = "Run an exported YOLOv8 ONNX model and print detections as JSON"
)]
struct Args {
    /// Path to the ONNX model (export with scripts/export_yolo_onnx.py)
    #[arg(long)]
    model: PathBuf,

    /// Image to run detection on; if omitted, runs one zero tensor as a smoke test
    #[arg(long)]
    image: Option<PathBuf>,

    /// Square model input size used at export time
    #[arg(long, default_value_t = 640)]
    size: u32,

    /// Minimum class score to keep a detection (configs/default.yaml: detection.confidence_threshold)
    #[arg(long, default_value_t = 0.5)]
    conf: f32,

    /// IoU above which same-class boxes are suppressed (configs/default.yaml: detection.nms_threshold)
    #[arg(long, default_value_t = 0.45)]
    iou: f32,
}

#[derive(Serialize)]
struct Report {
    image: String,
    width: u32,
    height: u32,
    detections: Vec<Detection>,
    timings_ms: Timings,
}

#[derive(Serialize)]
struct Timings {
    preprocess: f64,
    inference: f64,
    postprocess: f64,
}

fn main() -> Result<()> {
    // Logs go to stderr so stdout stays clean JSON.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    ort::init().with_name("vision-inference").commit();

    let mut session = Session::builder()?
        .commit_from_file(&args.model)
        .with_context(|| format!("load ONNX model from {:?}", args.model))?;

    match &args.image {
        Some(path) => detect(&mut session, path, &args),
        None => smoke_run(&mut session, args.size),
    }
}

fn detect(session: &mut Session, path: &Path, args: &Args) -> Result<()> {
    let image = image::open(path)
        .with_context(|| format!("open {path:?}"))?
        .to_rgb8();

    let started = Instant::now();
    let (input, letterbox) = preprocess(&image, args.size);
    let preprocessed = Instant::now();

    let outputs = session
        .run(ort::inputs![
            Tensor::from_array(input.into_dyn()).context("build input tensor")?
        ])
        .context("ONNX Runtime session.run")?;
    let inferred = Instant::now();

    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("model output 0 is not an f32 tensor")?;
    let dims = shape
        .iter()
        .map(|&d| usize::try_from(d))
        .collect::<Result<Vec<_>, _>>()
        .context("model output has a negative dimension")?;
    let mut detections = nms(decode_yolov8(&dims, data, args.conf)?, args.iou);
    for det in &mut detections {
        let (x1, y1) = letterbox.to_original(det.bbox[0], det.bbox[1]);
        let (x2, y2) = letterbox.to_original(det.bbox[2], det.bbox[3]);
        det.bbox = [x1, y1, x2, y2];
    }
    let postprocessed = Instant::now();

    let ms = |from: Instant, to: Instant| (to - from).as_secs_f64() * 1000.0;
    let report = Report {
        image: path.display().to_string(),
        width: image.width(),
        height: image.height(),
        detections,
        timings_ms: Timings {
            preprocess: ms(started, preprocessed),
            inference: ms(preprocessed, inferred),
            postprocess: ms(inferred, postprocessed),
        },
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn smoke_run(session: &mut Session, size: u32) -> Result<()> {
    tracing::info!("no --image; running one zero tensor shaped (1, 3, {size}, {size})");
    let input = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    let outputs = session
        .run(ort::inputs![
            Tensor::from_array(input.into_dyn()).context("build input tensor")?
        ])
        .context("ONNX Runtime session.run")?;

    tracing::info!("output tensors: {}", outputs.len());
    for (i, (name, value)) in outputs.iter().enumerate() {
        match value.try_extract_tensor::<f32>() {
            Ok((shape, _)) => tracing::info!("  output[{i}] ({name}): shape {shape:?}"),
            Err(_) => tracing::info!("  output[{i}] ({name}): not an f32 tensor"),
        }
    }
    Ok(())
}

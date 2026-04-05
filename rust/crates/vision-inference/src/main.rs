use anyhow::{Context, Result};
use clap::Parser;
use image::imageops::FilterType;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "vision-inference",
    about = "ONNX Runtime node for exported YOLO / vision models (Rust inference path)"
)]
struct Args {
    /// Path to ONNX model (export from Python with scripts/export_yolo_onnx.py)
    #[arg(long)]
    model: PathBuf,

    /// Optional image; if omitted, runs a single dummy tensor through the model
    #[arg(long)]
    image: Option<PathBuf>,

    /// Letterbox square size (typical YOLO export: 640)
    #[arg(long, default_value_t = 640_u32)]
    size: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    ort::init()
        .with_name("vision-inference")
        .commit();

    let mut session = Session::builder()?
        .commit_from_file(&args.model)
        .with_context(|| format!("load ONNX model from {:?}", args.model))?;

    let input = if let Some(path) = &args.image {
        preprocess_image(path, args.size).context("preprocess image")?
    } else {
        tracing::info!("no --image; running zeros tensor shaped for common YOLO ONNX (1,3,640,640)");
        Array4::<f32>::zeros((1, 3, args.size as usize, args.size as usize))
    };

    let input_tensor = Tensor::from_array(input.into_dyn())
        .context("build input tensor")?;
    let outputs = session
        .run(ort::inputs![input_tensor])
        .context("ONNX Runtime session.run")?;

    tracing::info!("output tensors: {}", outputs.len());
    for (i, (name, value)) in outputs.iter().enumerate() {
        match value.try_extract_tensor::<f32>() {
            Ok((shape, _data)) => tracing::info!("  output[{}] ({}): shape {:?}", i, name, shape),
            Err(_) => tracing::info!("  output[{}] ({}): (could not extract as f32 tensor)", i, name),
        }
    }

    Ok(())
}

/// RGB, float32, NCHW, values in [0, 1], letterboxed to `size` x `size`.
fn preprocess_image(path: &PathBuf, size: u32) -> Result<Array4<f32>> {
    let img = image::open(path).with_context(|| format!("open {:?}", path))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let scale = (size as f32 / w as f32).min(size as f32 / h as f32);
    let nw = (w as f32 * scale).round() as u32;
    let nh = (h as f32 * scale).round() as u32;
    let resized = image::imageops::resize(&rgb, nw, nh, FilterType::Triangle);
    let mut canvas = image::RgbImage::new(size, size);
    let ox = (size - nw) / 2;
    let oy = (size - nh) / 2;
    for y in 0..nh {
        for x in 0..nw {
            canvas.put_pixel(ox + x, oy + y, *resized.get_pixel(x, y));
        }
    }
    let mut arr = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    for y in 0..size {
        for x in 0..size {
            let p = canvas.get_pixel(x, y);
            arr[[0, 0, y as usize, x as usize]] = p[0] as f32 / 255.0;
            arr[[0, 1, y as usize, x as usize]] = p[1] as f32 / 255.0;
            arr[[0, 2, y as usize, x as usize]] = p[2] as f32 / 255.0;
        }
    }
    Ok(arr)
}

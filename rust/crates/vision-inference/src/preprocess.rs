//! Letterbox preprocessing that matches Ultralytics' `LetterBox` transform.
//!
//! YOLO models are trained on letterboxed inputs: the image is scaled to fit a square while
//! keeping its aspect ratio, then centered on a gray (114) canvas. Feeding the model a
//! differently padded image (for example black padding) silently costs accuracy, so this mirrors
//! `ultralytics/data/augment.py::LetterBox` defaults: OpenCV-style bilinear resize, centered,
//! pad value 114, and the `round(d - 0.1)` / `round(d + 0.1)` split for odd padding.

use image::{Rgb, RgbImage};
use ndarray::Array4;

/// Ultralytics' default letterbox pad value.
pub const PAD_VALUE: u8 = 114;

/// How an original image was placed on the square model input.
///
/// Needed to map detections from model-input pixels back to original-image pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// Side length of the square model input.
    pub size: u32,
    /// Scale applied to the original image (`resized = original * scale`).
    pub scale: f32,
    /// Columns of padding on the left.
    pub pad_left: u32,
    /// Rows of padding on the top.
    pub pad_top: u32,
    /// Original image width in pixels.
    pub orig_width: u32,
    /// Original image height in pixels.
    pub orig_height: u32,
}

impl Letterbox {
    /// Compute the placement of a `width` x `height` image on a `size` x `size` canvas.
    pub fn new(width: u32, height: u32, size: u32) -> Self {
        let scale = (size as f32 / width as f32).min(size as f32 / height as f32);
        let new_w = (width as f32 * scale).round() as u32;
        let new_h = (height as f32 * scale).round() as u32;
        let dw = (size - new_w) as f32 / 2.0;
        let dh = (size - new_h) as f32 / 2.0;
        Self {
            size,
            scale,
            pad_left: (dw - 0.1).round().max(0.0) as u32,
            pad_top: (dh - 0.1).round().max(0.0) as u32,
            orig_width: width,
            orig_height: height,
        }
    }

    /// Width and height of the image after scaling, before padding.
    pub fn resized_dims(&self) -> (u32, u32) {
        (
            (self.orig_width as f32 * self.scale).round() as u32,
            (self.orig_height as f32 * self.scale).round() as u32,
        )
    }

    /// Map a point from model-input pixels to original-image pixels, clamped to the image.
    pub fn to_original(&self, x: f32, y: f32) -> (f32, f32) {
        let ox = (x - self.pad_left as f32) / self.scale;
        let oy = (y - self.pad_top as f32) / self.scale;
        (
            ox.clamp(0.0, self.orig_width as f32),
            oy.clamp(0.0, self.orig_height as f32),
        )
    }
}

/// Letterbox `image` onto a gray `size` x `size` canvas.
pub fn letterbox_image(image: &RgbImage, size: u32) -> (RgbImage, Letterbox) {
    let lb = Letterbox::new(image.width(), image.height(), size);
    let (new_w, new_h) = lb.resized_dims();
    let resized = resize_bilinear(image, new_w, new_h);
    let mut canvas = RgbImage::from_pixel(size, size, Rgb([PAD_VALUE; 3]));
    image::imageops::replace(&mut canvas, &resized, lb.pad_left.into(), lb.pad_top.into());
    (canvas, lb)
}

/// Resize like OpenCV's `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`, which Ultralytics uses.
///
/// Each output pixel samples the 2x2 nearest source pixels at
/// `src = (dst + 0.5) * (src_size / dst_size) - 0.5`, clamped to the image. Unlike the `image`
/// crate's `Triangle` filter, the kernel does not widen when downscaling, so the model sees the
/// same (unblurred) pixels it was trained on.
pub fn resize_bilinear(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (src_w, src_h) = image.dimensions();
    let axis = |dst: u32, src_len: u32, dst_len: u32| {
        let pos = ((dst as f32 + 0.5) * (src_len as f32 / dst_len as f32) - 0.5).max(0.0);
        let i0 = (pos.floor() as u32).min(src_len - 1);
        let i1 = (i0 + 1).min(src_len - 1);
        (i0, i1, pos - i0 as f32)
    };
    let mut out = RgbImage::new(width, height);
    for y in 0..height {
        let (y0, y1, fy) = axis(y, src_h, height);
        for x in 0..width {
            let (x0, x1, fx) = axis(x, src_w, width);
            let (p00, p10) = (image.get_pixel(x0, y0), image.get_pixel(x1, y0));
            let (p01, p11) = (image.get_pixel(x0, y1), image.get_pixel(x1, y1));
            let mut px = [0u8; 3];
            for c in 0..3 {
                let top = f32::from(p00[c]) * (1.0 - fx) + f32::from(p10[c]) * fx;
                let bottom = f32::from(p01[c]) * (1.0 - fx) + f32::from(p11[c]) * fx;
                px[c] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, Rgb(px));
        }
    }
    out
}

/// Convert an RGB image to an NCHW float tensor with values in `[0, 1]`.
pub fn to_nchw(image: &RgbImage) -> Array4<f32> {
    let (w, h) = image.dimensions();
    let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    for (x, y, pixel) in image.enumerate_pixels() {
        for c in 0..3 {
            tensor[[0, c, y as usize, x as usize]] = f32::from(pixel[c]) / 255.0;
        }
    }
    tensor
}

/// Letterbox and convert an image in one step, returning the model input and its placement.
pub fn preprocess(image: &RgbImage, size: u32) -> (Array4<f32>, Letterbox) {
    let (canvas, lb) = letterbox_image(image, size);
    (to_nchw(&canvas), lb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landscape_image_is_padded_top_and_bottom() {
        let lb = Letterbox::new(200, 100, 64);
        assert!((lb.scale - 0.32).abs() < 1e-6);
        assert_eq!(lb.resized_dims(), (64, 32));
        assert_eq!(lb.pad_left, 0);
        assert_eq!(lb.pad_top, 16);
    }

    #[test]
    fn odd_padding_puts_the_extra_row_at_the_bottom() {
        // 64 - 31 = 33 rows of padding: Ultralytics puts 16 on top and 17 on the bottom.
        let lb = Letterbox::new(64, 31, 64);
        assert_eq!(lb.resized_dims(), (64, 31));
        assert_eq!(lb.pad_top, 16);
    }

    #[test]
    fn padding_is_gray_114_and_image_is_centered() {
        let img = RgbImage::from_pixel(200, 100, Rgb([255, 0, 0]));
        let (canvas, lb) = letterbox_image(&img, 64);
        assert_eq!(canvas.dimensions(), (64, 64));
        assert_eq!(canvas.get_pixel(32, 0), &Rgb([PAD_VALUE; 3]));
        assert_eq!(canvas.get_pixel(32, 63), &Rgb([PAD_VALUE; 3]));
        assert_eq!(canvas.get_pixel(32, lb.pad_top + 16), &Rgb([255, 0, 0]));
    }

    /// Gray image from a row-major grid of values.
    fn gray(width: u32, values: &[u8]) -> RgbImage {
        let height = values.len() as u32 / width;
        RgbImage::from_fn(width, height, |x, y| {
            Rgb([values[(y * width + x) as usize]; 3])
        })
    }

    fn row_major(image: &RgbImage) -> Vec<u8> {
        image.pixels().map(|p| p[0]).collect()
    }

    // Expected values come from cv2.resize(..., interpolation=cv2.INTER_LINEAR).
    #[test]
    fn resize_matches_opencv_inter_linear() {
        let down = resize_bilinear(&gray(4, &[0, 100, 200, 250]), 2, 1);
        assert_eq!(row_major(&down), vec![50, 225]);

        let up = resize_bilinear(&gray(2, &[0, 100]), 4, 1);
        assert_eq!(row_major(&up), vec![0, 25, 75, 100]);

        let grid: Vec<u8> = (0..25).map(|v| v * 10).collect();
        let down_2d = resize_bilinear(&gray(5, &grid), 3, 3);
        assert_eq!(
            row_major(&down_2d),
            vec![20, 37, 53, 103, 120, 137, 187, 203, 220]
        );
    }

    #[test]
    fn tensor_is_nchw_and_normalized() {
        let img = RgbImage::from_pixel(4, 2, Rgb([255, 0, 51]));
        let tensor = to_nchw(&img);
        assert_eq!(tensor.shape(), &[1, 3, 2, 4]);
        assert!((tensor[[0, 0, 1, 3]] - 1.0).abs() < 1e-6);
        assert!(tensor[[0, 1, 1, 3]].abs() < 1e-6);
        assert!((tensor[[0, 2, 1, 3]] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn model_coordinates_map_back_to_the_original_image() {
        let lb = Letterbox::new(200, 100, 64);
        // Model-input (32, 32) is the center of the canvas, so the center of the original.
        let (x, y) = lb.to_original(32.0, 32.0);
        assert!((x - 100.0).abs() < 1e-3);
        assert!((y - 50.0).abs() < 1e-3);
        // Points in the padding clamp to the image edge.
        assert_eq!(lb.to_original(0.0, 0.0), (0.0, 0.0));
    }
}

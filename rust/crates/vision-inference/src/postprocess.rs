//! Decode YOLOv8 ONNX output into detections and run non-maximum suppression.
//!
//! An Ultralytics YOLOv8 detection export emits one tensor shaped `(1, 4 + nc, N)`: for each of
//! the `N` anchors, `cx, cy, w, h` in model-input pixels followed by `nc` sigmoid class scores.
//! There is no separate objectness channel (unlike YOLOv5), so an anchor's confidence is its best
//! class score.

use anyhow::{bail, ensure, Result};
use serde::Serialize;

/// Upper bound on detections returned per image, matching Ultralytics' `max_det` default.
pub const MAX_DETECTIONS: usize = 300;

/// One detected object.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Detection {
    pub class_id: usize,
    pub score: f32,
    /// `[x1, y1, x2, y2]` in pixels.
    pub bbox: [f32; 4],
}

/// Decode a YOLOv8 output tensor, keeping anchors whose best class score is at least `conf`.
///
/// `shape` and `data` are the tensor's dimensions and row-major values. Boxes are returned in
/// model-input pixels; map them back with [`crate::preprocess::Letterbox::to_original`].
pub fn decode_yolov8(shape: &[usize], data: &[f32], conf: f32) -> Result<Vec<Detection>> {
    let [batch, channels, anchors] = shape else {
        bail!("expected a 3-D output (1, 4 + classes, anchors), got shape {shape:?}");
    };
    let (channels, anchors) = (*channels, *anchors);
    ensure!(*batch == 1, "expected batch size 1, got {batch}");
    ensure!(
        channels > 4,
        "expected at least one class channel after the 4 box channels, got {channels} channels"
    );
    ensure!(
        data.len() == channels * anchors,
        "output has {} values but shape {shape:?} needs {}",
        data.len(),
        channels * anchors
    );

    let at = |c: usize, i: usize| data[c * anchors + i];
    let mut detections = Vec::new();
    for i in 0..anchors {
        let (class_id, score) =
            (4..channels)
                .map(|c| (c - 4, at(c, i)))
                .fold(
                    (0, f32::MIN),
                    |best, cur| if cur.1 > best.1 { cur } else { best },
                );
        if score < conf {
            continue;
        }
        let (cx, cy, w, h) = (at(0, i), at(1, i), at(2, i), at(3, i));
        detections.push(Detection {
            class_id,
            score,
            bbox: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        });
    }
    Ok(detections)
}

/// Intersection over union of two `[x1, y1, x2, y2]` boxes.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter_w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let inter_h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = inter_w * inter_h;
    let area = |r: &[f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let union = area(a) + area(b) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Per-class non-maximum suppression, highest score first, capped at [`MAX_DETECTIONS`].
///
/// A box is dropped when it overlaps an already-kept box of the same class by more than
/// `iou_threshold`. Boxes of different classes never suppress each other.
pub fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for det in detections {
        let suppressed = kept
            .iter()
            .any(|k| k.class_id == det.class_id && iou(&k.bbox, &det.bbox) > iou_threshold);
        if !suppressed {
            kept.push(det);
            if kept.len() == MAX_DETECTIONS {
                break;
            }
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `(1, 4 + nc, N)` tensor from per-anchor rows of `[cx, cy, w, h, scores...]`.
    fn tensor(rows: &[Vec<f32>]) -> (Vec<usize>, Vec<f32>) {
        let channels = rows[0].len();
        let anchors = rows.len();
        let mut data = vec![0.0; channels * anchors];
        for (i, row) in rows.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                data[c * anchors + i] = *v;
            }
        }
        (vec![1, channels, anchors], data)
    }

    #[test]
    fn decode_keeps_confident_anchors_with_their_best_class() {
        let (shape, data) = tensor(&[
            vec![50.0, 40.0, 20.0, 10.0, 0.10, 0.90],
            vec![10.0, 10.0, 4.0, 4.0, 0.20, 0.30],
        ]);
        let dets = decode_yolov8(&shape, &data, 0.5).unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 1);
        assert!((dets[0].score - 0.9).abs() < 1e-6);
        assert_eq!(dets[0].bbox, [40.0, 35.0, 60.0, 45.0]);
    }

    #[test]
    fn decode_rejects_unexpected_shapes() {
        assert!(decode_yolov8(&[1, 84], &[0.0; 84], 0.5).is_err());
        assert!(decode_yolov8(&[2, 6, 1], &[0.0; 12], 0.5).is_err());
        assert!(decode_yolov8(&[1, 4, 3], &[0.0; 12], 0.5).is_err());
        assert!(decode_yolov8(&[1, 6, 2], &[0.0; 5], 0.5).is_err());
    }

    #[test]
    fn iou_of_identical_disjoint_and_half_overlapping_boxes() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
        assert_eq!(iou(&a, &[20.0, 20.0, 30.0, 30.0]), 0.0);
        // Overlap 50 of a 150 union.
        assert!((iou(&a, &[5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn nms_suppresses_same_class_overlaps_only() {
        let det = |class_id, score, bbox| Detection {
            class_id,
            score,
            bbox,
        };
        let kept = nms(
            vec![
                det(0, 0.6, [1.0, 1.0, 11.0, 11.0]),
                det(0, 0.9, [0.0, 0.0, 10.0, 10.0]),
                det(1, 0.8, [0.0, 0.0, 10.0, 10.0]),
                det(0, 0.7, [50.0, 50.0, 60.0, 60.0]),
            ],
            0.45,
        );
        let scores: Vec<f32> = kept.iter().map(|d| d.score).collect();
        assert_eq!(scores, vec![0.9, 0.8, 0.7]);
    }
}

# ADR 001: Keypoint Pipeline vs RGB-D Fusion

## Status

Proposed

## Context

The project needs a 6-DoF object pose estimator that is credible for a portfolio, measurable on YCB-style objects, and realistic to build in a 12-16 week schedule.

## Options

- Keypoint-based pose estimation: predict 2D keypoints, then solve PnP with known 3D object geometry.
- RGB-D fusion: use depth directly in a DenseFusion-style model.

## Decision

Start with the keypoint-based pipeline unless early experiments show depth is necessary for the selected objects.

## Consequences

- Stronger portfolio signal than a detector-only baseline.
- Clear geometry story through PnP and known object models.
- Easier to debug with overlays and failed keypoints.
- Depth can still be used later for refinement or ablation.

# Toss-to-bin plan

Decision record: [ADR 003](decisions/003-toss-to-bin.md). This is a Phase 7 extension and the
first thing cut if the schedule slips.

## Loop

```
camera RGB-D ──► YOLO (ball, bin boxes) ──► back-project box centre + depth ──► bin xyz (world)
                                                                                    │
grasped ball at release point p0 ◄─────────── arm (Franka, RMPflow) ◄───────────────┤
                                                                                    ▼
                         release velocity v = v_ballistic(p0, bin, θ) + residual(features)
                                                                                    │
                                                          throw, then score: in bin or not
```

## Slices

Each slice is one PR. Slices 1-3 need no simulator and are tested in CI.

| # | Slice | Needs Isaac Sim | Done when |
|---|-------|-----------------|-----------|
| 1 | `isaacx.throwing.ballistics`: closed-form release speed for a fixed release angle, forward integrator with optional quadratic drag | No | Round-trip test: the solved speed lands inside the bin rim with drag off; unreachable targets are rejected; with drag on, the same speed falls short |
| 2 | `isaacx.perception.backproject`: pixel + depth to a camera-frame point, camera-to-world transform | No | Tests against hand-computed points and a known extrinsic |
| 3 | Residual model + toy evaluation: ridge regression on simulated throws with injected drag and release bias | No | Hit rate with residual vs physics-only on held-out targets, reported as a toy result |
| 4 | Isaac Sim throwing scene: Franka, bins, balls, release at a commanded velocity | Yes | Scene runs headless on the workstation and logs landing points |
| 5 | Perception in the loop: YOLO bin detection + slice 2 in the scene | Yes | Bin position error vs ground truth reported |
| 6 | Closed loop and evaluation: grasp, throw, score over N trials | Yes | Hit-rate table in `docs/results/`, physics-only vs residual |

## Ground rules

- Soft objects into bins only. Never people, animals or fragile things as targets.
- Every number says where it came from: toy simulator, Isaac Sim, or real hardware.
- Release angle is fixed per throw (45 degrees to start), so the solver has one unknown: speed.

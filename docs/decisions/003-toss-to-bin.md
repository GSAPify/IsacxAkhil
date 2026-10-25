# ADR 003: Toss-to-bin as the End-to-End Task

## Status

Proposed

## Context

The pose stack (Phases 1-6) ends at "the model knows where the object is". A portfolio piece
reads better when that output drives an action. Throwing is a good action to show because it
needs the full loop: see the object, see the target, grasp, and release at the right velocity.
The result is also easy to score: in the bin or not.

The task is bounded and non-hazardous by design. Soft objects from the YCB set (`056_tennis_ball`,
`057_racquetball`, `061_foam_brick`) are thrown into fixed bins (`024_bowl` or a simple box)
inside the arm's workspace. Targets are always bins, never people or animals.

The approach is inspired by TossingBot (Zeng et al., 2019): estimate a release velocity from
physics, then learn a residual that corrects for what physics does not model. This repo does not
claim to reproduce TossingBot's training setup or results.

## Options

- Keep the project at pose estimation and tracking, with no action.
- Pick-and-place as the action. Simple, but it does not stress timing or calibration.
- Toss-to-bin as the action: ballistic release planning plus a learned residual.

## Decision

Toss-to-bin is a **Phase 7 extension** that consumes the Phase 5 real-time loop. It does not
change Phases 1-6 or their order.

The parts that need no simulator (ballistics, back-projection from depth, the residual and its
toy evaluation) are pure Python with tests. They can land early because they cost little and do
not block the Isaac Sim timeline. The parts that need Isaac Sim (throwing scene, grasp, release
control) wait until Phase 5 works.

In the existing cut order, the toss extension is cut **first**, before full Rust inference.
Sim-to-real photos, latency and failure analysis are still never cut.

## Consequences

- The project gets a visible end-to-end demo without moving the core perception milestones.
- Residual numbers from the toy simulator are reported as toy results only, never as robot
  results.
- Real-world throwing is out of scope until the sim results are solid, and needs a padded,
  enclosed workspace when it happens.

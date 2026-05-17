# ADR 002: Rust Scope

## Status

Proposed

## Context

The project should demonstrate systems judgment without letting Rust integration consume the core perception timeline.

## Options

- Minimal scope: implement the tracker as a standalone Rust crate and keep model inference in Python/ONNX.
- Ambitious scope: run full ONNX inference in Rust with `ort` or another runtime.

## Decision

Default to the minimal Rust tracker scope. Revisit full Rust inference only after the model, evaluation, and latency measurements are working.

## Consequences

- Keeps the Rust component tied to real-time state estimation rather than becoming a tooling detour.
- Preserves schedule for sim-to-real evaluation and failure analysis.
- Leaves room for full Rust inference as an extension if the core project lands early.

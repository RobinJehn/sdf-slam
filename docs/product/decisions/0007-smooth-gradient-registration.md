---
id: 0007
title: Smooth-gradient registration recipe for real data
type: architecture
owner: Robin Jehn
created: 2026-09-13
updated: 2026-09-13
requirements:
- ../PRD.md
---

# DEC-0007 — Smooth-gradient registration recipe for real data

## Context

The full-Intel reconstruction was stretched ~20% along the corridor loop and no parameter change fixed it. The original repo's saved artifacts (a 193-scan, 150x150 run matching dissertation fig. 4.18) scored a 0.378 m median revisit error; our reimplementation scored 0.753 on the same slice.

## Decision

The Intel configs use a four-part recipe; each part alone does not work (best single-change score: 0.44 m):

- `smooth_gradient: true` — the pose Jacobian of point residuals uses the central-difference SDF gradient interpolated at the point. The exact bilinear-patch gradient is discontinuous at cell borders and kinks the registration force; the smooth gradient keeps it continuous.
- `normals: pca, k_neighbors: 7` — matches the original's normal estimation for this dataset.
- `active_region: false` with `eikonal: 0.04` — an Eikonal residual at every cell propagates the SDF beyond observed walls, so scans that reach unmapped cells still see a meaningful gradient. The weight matches the original's convention (their factor scales the residual linearly; 0.2 there equals 0.04 in our squared-weight convention).
- `lambda_init: 5, lambda_factor: 0.9, step_tolerance: 0.001` — the original's damping schedule: lambda grows on success, so late iterations take small careful steps.

Benchmark (median revisit-consistency, `tools/viz/revisit_consistency.py`): lap-1 slice 0.013 m vs the original artifacts' 0.378 m; full 910 scans 0.065 m vs 0.540 m for the previous best config. The corridor stretch is gone (robust extent 35.6 x 32.9 m).

## Alternatives considered

- Exact bilinear-patch gradient (tried, reverted 2026-09-13): analytically correct and it passes finite-difference tests, but registration quality collapses on real data.
- Trust-region LM, Huber loss, Marquardt scaling, per-scan normals, batch polish (tried 2026-09-13): none closed the gap; kept as options where implemented.
- Active-region state with the smooth-gradient recipe: incompatible, since the global Eikonal field needs every node in the state. Active region stays available for speed-focused runs (DEC-0002).

## Consequences

- Full-Intel runs optimize the dense state: ~42 min at 100x100 instead of ~16 min with the active region.
- The pose Jacobian is deliberately inconsistent with the residual's true derivative; finite-difference Jacobian tests cover the default (exact) mode only.
- DEC-0003's `weighted` normals stay the default for the simulated datasets; the Intel configs override to `pca`.

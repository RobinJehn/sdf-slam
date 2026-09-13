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

- The recipe is resolution-sensitive as recorded: naive 200x200 fails
  (0.670 vs 0.065). `smooth_gradient_step` (meters) plus an eikonal weight
  scaled by cell area recovers most of it at 200x200 (0.117, thinner error
  tail, 0.27 m cells) but does not beat the 100x100 flagship; 100x100 with
  the recorded values stays the reference configuration.

- `smooth_gradient` is on by default: on the simulated dataset with ground
  truth it halves the mean translation error (0.039 to 0.022) and matches the
  dissertation's table 4.2. The exact bilinear-patch gradient stays available
  as `smooth_gradient: false`.
- Huber loss and the trust-region mode also fail ON TOP of the recipe
  (lap-1: 0.69 / 0.56 vs 0.013), so the negative results hold in both
  directions for those two.
- The odometry weight stays at 1.0 (sweep 2026-09-13). The term is
  load-bearing: weight 0 and weight 2 both diverge on lap-1 (0.822 /
  0.906 vs 0.013). Weight 0.5 wins on lap-1 (0.012, p90 0.068) but
  diverges on the full dataset (0.411 vs 0.065 NN; 3.1 m vs 0.057 m
  median ICP-relations error; ghosted map) — another case where the
  lap-1 slice does not proxy the full box.

- Full-Intel runs optimize the dense state: ~42 min at 100x100 instead of ~16 min with the active region.
- The pose Jacobian is deliberately inconsistent with the residual's true derivative; finite-difference Jacobian tests cover the default (exact) mode only.
- DEC-0003's `weighted` normals stay the default for the simulated datasets; the Intel configs override to `pca`.

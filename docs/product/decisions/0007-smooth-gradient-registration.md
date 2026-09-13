---
id: 0007
title: Smooth-gradient registration recipe for real data
type: architecture
owner: Robin Jehn
created: 2026-09-13
updated: 2026-09-14
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

- Resolution transfer is solved by the stencil identity (2026-09-14). The
  default smooth gradient interpolates node central differences, so it is
  exact per axis at every resolution. The `smooth_gradient_step` meters
  knob reproduces it only at exactly one cell (h = extent/(nx-1)); other
  values land in an unpredictable basin patchwork (lap-1 ensembles: ratio
  0.95/1.05/1.25 uniformly bad, 1.10 flips 2/6, 1.50 good — no monotone
  law). Rule: use the default stencil; the meters knob is for controlled
  smoothing-width experiments only.
- With the default stencil and an area-scaled eikonal weight
  (0.04 x (h/0.5354)^2), the recipe transfers across resolutions:
  150x150 matches the 100x100 flagship on full Intel (ICP-relations
  median 0.052-0.057 m over three lambda-jittered runs, NN 0.080-0.093,
  thinner tail than 100x100). 200x200 (h=0.27 m) converges but degrades
  (0.095): a resolution floor exists near h~0.25 m for this dataset —
  an anchored lap-1 config at h=0.20 m is uniformly bad (8/8 at
  0.67-0.93). Earlier single-run sweeps at 150x150 and 200x200 that
  motivated other conclusions are superseded; single runs in this regime
  are draws from a basin distribution and rank differently under the NN
  and ICP-relations metrics (compression bias).
- Stability is measurable, not assumed (`tools/viz/stability_ensemble.py`,
  8 lambda-jittered runs). Anchored configs (default stencil, h in
  0.27-0.54 m, area-scaled eikonal) score 8/8 converged with ~zero spread
  on every testbed tried; the original lap-1 config with unscaled eikonal
  0.04 at h~0.25 m flips 2/8 — the area scaling is a basin-widener, not
  only an accuracy tweak. Off-anchor points are unpredictable, and
  `reject_worse_steps` does not rescue them.
- Residual windows are narrow (lap-1 anchored ensembles): hallucination
  off or points 3/12 instead of 6 is uniformly bad; hallucination weight
  0.5-1.0 is fine; eikonal x0.5 is bad and x2 flips 4/6. Choose new
  configurations by the anchored rules, then gate on an ensemble
  (see DEC-0009).
- The 1e-9 `lambda_init` chaos (DEC-0008) is config-dependent: it appears
  near basin boundaries (original lap-1 eikonal 0.04: 2/8 diverge) and
  vanishes deep inside anchored basins (spread 0.000). Performance work
  must stay bitwise-exact because production configs are not guaranteed
  to sit deep in a basin.
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

---
id: 0009
title: Choose parameters by anchored rules plus stability ensembles
type: process
owner: Robin Jehn
created: 2026-09-14
updated: 2026-09-14
requirements:
- ../PRD.md
---

# DEC-0009 — Choose parameters by anchored rules plus stability ensembles

## Context

The recipe (DEC-0007) sits on a multi-dimensional equilibrium. The parameter
landscape is a deterministic patchwork of basins: nearby values flip between
converged (~0.06 m) and diverged (2-20 m), the pattern is non-monotone along
every tested axis, and some points flip under a 1e-9 lambda jitter
(DEC-0008). A single run therefore cannot rank configurations, and no
closed-form rule alone is safe. Intel has no ground truth, so the metric
itself also needs care: the nearest-neighbor revisit metric rewards
compressed estimates and ranks diverged runs incorrectly.

## Decision

Configure a new dataset or resolution in three steps:

1. **Anchor.** Compute candidates from the scaling rules instead of tuning
   freely: default smooth-gradient stencil (never the meters knob for
   production); cell size h between 0.27 and 0.54 m for indoor lidar
   (below 0.27 m the response is patchwork — the gate decides),
   h = extent/(nx-1); eikonal weight 0.04 x (h/0.5354)^2; the other recipe
   values as recorded in DEC-0007.
2. **Gate on an ensemble.** Run the candidate 8 times with 1e-9-scale
   `lambda_init` jitters (`tools/viz/stability_ensemble.py`) on a cheap
   slice. Accept only 8/8 converged with near-zero spread. A flip rate > 0
   means the config sits near a basin boundary and will not survive
   compiler, BLAS, or dataset changes.
3. **Validate with relative-pose references.** Score full runs against
   frozen ICP relations (`tools/viz/icp_relations.py`: point-to-line ICP on
   revisit pairs, with residual, observability, and seed-deviation guards)
   plus the rendered map. Never rank runs by the NN metric alone.

Every anchored config passed the ensemble gate on every testbed tried;
every off-anchor config was uniformly bad, flip-prone, or lucky. The rules
supply good candidates; only the ensemble certifies them.

## Alternatives considered

- Single-run scoring with the NN revisit metric (tried, superseded
  2026-09-14: single runs are basin draws; the NN metric scored a fully
  collapsed map 0.096, better than the flagship, while ICP relations put it
  at 12.3 m).
- Deterministic parameter laws without a gate (tried, insufficient
  2026-09-14: an "s = k x h stencil law" and a monotone eikonal-scaling law
  both failed controlled ensembles; the landscape is non-monotone along
  every axis).
- Stabilizing the solver so parameters stop mattering
  (`reject_worse_steps`, tried 2026-09-14: does not rescue off-anchor
  configs; eikonal x2 and lambda 20 rescues also failed).

## Evidence

- 62d6c53 (2026-09-13, Intel 910 scans): ICP-relations evaluator; 189
  frozen pairs; flagship 0.057 m vs diverged runs 12-19 m.
- 2eca082 (2026-09-13/14, lap-1 square box, 8 jitters per config): anchored
  configs at h = 0.268/0.404 m and anisotropic h ~ 0.25 m with scaled
  eikonal: 8/8, spread 0.000. Original eikonal 0.04 at the same grid: 2/8
  diverged. Stencil ratios 0.95/1.05/1.25: uniformly bad; 1.10: 2/6 flips.
  Anchored h = 0.20 m: uniformly bad (resolution floor).
- Full Intel 150x150 anchored (2026-09-14, four runs incl. three
  lambda-jittered): ICP-relations median 0.052-0.061 m — reproducible.
- Full Intel 100x100 (2026-09-14, three lambda-jittered replicates):
  11.4 / 0.37 / 0.31 m vs the single-draw 0.057 m — the procedure caught
  the project's own headline config sitting on a basin boundary; the
  150x150 variant replaced it as the reference.
- Residual windows (lap-1 ensembles, 2026-09-14): hallucination off, points
  3 or 12, eikonal x0.5 all uniformly bad; hallucination weight 0.5 fine;
  eikonal x2 flips 4/6.
- Warm batch polish is parameter-insensitive (2026-09-14): two uniformly
  bad configs (stencil ratio 0.95: 0.72; hallucination points 12: 0.55)
  converge to the good solution when warm-started from a good trajectory
  (0.011 / 0.013). The narrow windows above apply to the incremental
  stage only; solver-knob stabilizers (iterations x2, tighter tolerance,
  lambda 10, lambda_factor 0.95) do NOT widen the incremental basin, and
  lambda_factor 0.95 makes it worse (5/6 diverged).

- End-to-end demo on simu_76_noise as a fresh environment (2026-09-14):
  the anchored config (h=0.5, eikonal 0.0349, recipe values, incremental)
  passes the gate 6/6 with zero spread and beats the tuned sweep runs on
  ground-truth relative errors (rel. translation 0.0029 vs 0.0043-0.0079)
  without any per-dataset tuning.

## Consequences

- Parameter work produces probabilities, not single numbers: results are
  reported as ensemble median + spread + flip count.
- The lap-1 slice is the standard cheap gate (~20 min per ensemble); full
  Intel runs only validate already-gated configs, with repeats.
- Dissertation-comparison claims must carry the fragility caveat: headline
  numbers are samples from a basin, valid for the exact binary and config.
- New-environment onboarding is a procedure (anchor, gate, validate), not
  a tuning session.

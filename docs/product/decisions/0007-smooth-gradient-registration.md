---
id: 0007
title: Smooth-gradient registration recipe for real data
type: architecture
owner: Robin Jehn
created: 2026-09-13
updated: 2026-09-15
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

The reference configuration is the 150x150 variant (`configs/intel_smooth_gradient_150.yaml`): it reproduces its score under lambda jitter (four full runs, ICP-relations median 0.052-0.061 m). The 100x100 config produced the 0.065/0.057 headline as a single draw but fails the DEC-0009 gate — three jittered replicates give 11.4 / 0.37 / 0.31 m — so its number is not reproducible and the config sits on a basin boundary. Use 100x100 only for speed-insensitive exploration.

A second stage completes the pipeline (2026-09-14): a batch polish at 200x200 warm-started from the reference trajectory (`configs/intel_polish_200.yaml`, ~60 s with CHOLMOD) improves the score to 0.050-0.054 m and is jitter-stable to the last digit — batch mode avoids the incremental chaining that makes the landscape chaotic. The polish also converges at 300x300: the incremental resolution patchwork does not bind in the warm batch regime, so resolution beyond the reference is a polish concern, not an incremental one. 200x200 is the sweet spot; ~0.05 m is likely the ICP-reference noise limit.

The reference runs 5 solver iterations per increment with the CHOLMOD linear solver (adopted 2026-09-15 per the DEC-0008 reordering-tier gate; ~17-18 min; certified 3/3 jittered runs at 0.053-0.055 m): halving the iterations from 10 loses no accuracy and halves the runtime. Increment size 5 instead of 1 diverges (9.7 m) — increments must stay small; iterations per increment are the cheap knob. The floor is 3 (2026-09-16, three jittered full runs each): 3 iterations score 0.051-0.056 m with healthy maps at ~10 min (45% faster than 5); 2 iterations diverge 3/3 (17-23 m). 5 stays the reference; 3 is the certified fast setting.

## Alternatives considered

- Exact bilinear-patch gradient (tried, reverted 2026-09-13): analytically correct and it passes finite-difference tests, but registration quality collapses on real data.
- Trust-region LM, Huber loss, Marquardt scaling, per-scan normals (tried 2026-09-13): none closed the gap; kept as options where implemented.
- Cold batch polish (tried 2026-09-13, revised 2026-09-14): from odometry it lands in a local minimum, but warm-started from the anchored reference trajectory it is the second pipeline stage (see Consequences).
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
  (0.095). Below h ~ 0.27 m the incremental response is patchwork, not a
  floor (lap-1 anchored ensembles: h=0.268 and 0.230 uniformly good,
  0.247 and 0.201 uniformly bad) — the DEC-0009 gate decides, and the
  warm batch polish is the reliable route to finer grids anyway.
  Earlier single-run sweeps at 150x150 and 200x200 that
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
  off is uniformly bad; the working geometry needs point spacing at or
  below 0.1 m AND a band at or below ~0.6 m (6 x 0.05 m: 6/6 at 0.009;
  3 x 0.1 m: uniformly bad; 6 x 0.2 m: flips 4/6; 12 x 0.1 m: uniformly
  bad). Hallucination weight 0.5-1.0 is fine; eikonal x0.5 is bad and x2
  flips 4/6. Choose new configurations by the anchored rules, then gate
  on an ensemble (see DEC-0009).
- The 1e-9 `lambda_init` chaos (DEC-0008) is config-dependent: it appears
  near basin boundaries (original lap-1 eikonal 0.04: 2/8 diverge) and
  vanishes deep inside anchored basins (spread 0.000). Performance work
  follows the DEC-0008 two-tier policy: refactors stay bitwise-exact,
  and arithmetic reorderings need the DEC-0009 gate per config because
  a config is not guaranteed to sit deep in a basin.
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
- Wheel odometry beats scan-matched odometry as the residual anchor
  (ensembles 2026-09-14): odometry-as-init-only is uniformly bad (6/6 at
  0.55-0.65), and replacing the wheel relative poses with point-to-line
  ICP odometry (`tools/viz/icp_odometry.py`) is also bad at loose and
  strict guards (5/6 and 4/6 bad). The scan-to-map residuals already do
  the matching inside the joint solve; pre-matched odometry injects
  correlated errors, while wheel noise is unbiased and averages out.
- ICP relations as ADDITIONAL residuals (loop closures) help
  (2026-09-14): `relations_file` + `weights.relation` add relative-pose
  residuals between revisit pairs (built by `tools/viz/icp_relations.py`
  from a previous run's estimate). On held-out pairs the stage-1 error
  tail nearly halves (mean 0.206 to 0.113 m at equal median); the
  polished result gains a little more (0.118 to 0.109). Certified over
  four lambda-jittered runs: held-out mean stays in 0.113-0.116 while
  the plain reference spreads 0.108-0.206 — the relations also collapse
  the tail variance, a third structural stabilizer besides the eikonal
  area scaling and the warm polish. Two-pass workflow: run once without
  relations, build relations from that estimate, rerun with them.
  Evaluate only on held-out pairs the constraints never saw.
- Relation gaps have a crossover between 2 and 5 (lap-1 ensembles,
  2026-09-14; `icp_relations.py build --min-gap/--max-gap` selects gap
  bands): a gap-2-only web is uniformly bad (8/8 at ~0.72 vs baseline
  0.011) — the same correlated-error mechanism that broke ICP odometry —
  while a gap-5 web and a mixed band [2,10] with 3 pairs per frame both
  pass 8/8 at baseline accuracy. The relation weight window is narrow:
  1.0 passes, 2.0 diverges 4/8, 5.0 diverges 7/8. On full Intel a dense
  band-[2,10] web (1649 pairs) scores median 0.053-0.055 vs the plain
  reference 0.055-0.062 on the frozen benchmark (three jittered runs,
  disjoint evidence) but does not collapse the tail variance (means
  0.119-0.137) and adds ~2.7x stage-1 runtime. Sparse loop-closure
  relations remain the better stabilizer per unit compute; the dense web
  is a certified but not recommended variant. The web also does not
  compose with loop closures (2026-09-15, three jittered runs each,
  CHOLMOD reference): on held-out pairs, loop closures alone score
  mean 0.110-0.111 while closures plus the dense web score 0.116-0.123
  with wider medians (0.057-0.065), a distorted map in the worst run,
  and ~2.9x runtime. Short-gap relations add correlated scan-match
  noise that fights the closures. Second-generation closures also fail
  (2026-09-15, three jittered runs): relations rebuilt from the
  closure-improved estimate pass the guards at 212 pairs (vs 95), but
  the held-out mean degrades to 0.112-0.121 and the medians spread —
  the marginal new pairs correlate with the estimate that admitted
  them, and the denser long-range coupling costs ~3.8x runtime. Build
  relations once, from the plain reference estimate. Per-relation
  residual weighting (G-ICP style, optional weight column in the CSV,
  `icp_relations.py build --weighted`) also loses to uniform weights
  (held-out mean 0.112-0.124 vs 0.110-0.111, three jittered runs each):
  the guards already truncate the residual distribution, so the
  remaining variation reflects scene geometry, not relation quality.
  Recommended pipeline stays: reference, optional loop-closure pass
  with uniform weights, warm polish.
- The full three-stage chain is certified end to end (2026-09-16, three
  jittered chains: reference run, closure rerun on its relations, warm
  200x200 polish from the closure trajectory): frozen benchmark median
  0.050-0.053 m with tail mean 0.110-0.116, held-out median 0.055-0.056
  with tail mean 0.108-0.113, healthy maps. The closure pass's
  tail-variance collapse survives the polish. Total runtime ~37 min
  with CHOLMOD (18 + 18 + 1).
- The anchor is exact but soft at range (2026-09-15, measured across
  jittered reference runs): frame 0 is hard-fixed, yet a near-global
  twist of {map, poses 1..N} costs only frame 0's scan residuals, so
  absolute pose deviation between runs grows with distance from the
  anchor (0.2-0.5 m near frame 0, 1-2 m at frames 600-900, 5-10 m on
  the post-last-revisit tail) while gap-1 relative deviation stays
  ~0.02 m. Run-to-run spread lives in this soft mode. This is inherent:
  absolute stiffness at frame j is the series stiffness of the chain
  from the anchor, and no metric in use scores absolute pose. Do not
  upweight frame 0 or fix extra frames; that injects artificial stress.
- Gauge decomposition of the run-to-run spread (2026-09-15, pairwise
  over the three plain and three closure jittered reference runs): a
  single best-fit global SE(2) transform removes almost none of the
  deviation (internal RMS is close to raw RMS in every pair), so the
  soft mode is differential, not a rigid lean. Loop closures stiffen
  the differential component only where relation pairs exist: inside
  the covered frames 14-681, the inter-run gap-200 relative deviation
  drops from 0.65 m mean / 1.57 m p95 (plain) to 0.44 / 1.12 (closures).
  The ICP relation set has zero pairs past frame 681, and the uncovered
  tail dominates the total spread; with closures that tail is no
  stiffer (gap-200 deviation 4.3 m plain vs 9.4 m closures in this
  three-run sample). Rule: closure stiffening is local to pair
  coverage — cover the whole trajectory with pairs, or expect the
  uncovered tail to keep the soft-mode spread. This also reconciles
  the held-out tail-variance collapse (held-out pairs live inside the
  covered region) with the large absolute run-to-run deviation.

- The converged map is not a distance field, and the objective prefers it
  that way (2026-09-16, simu batch runs). `map.init_from_scans` seeds every
  node with the signed distance to the nearest scan point instead of a
  constant. Against the constant-0 default it cuts whole-domain field error
  from 3.85-3.90 m to 0.86-2.15 m and lifts mean |grad map| from 0.22-0.34
  to 1.03-1.25, where a true signed distance field holds |grad| = 1. The
  default therefore converges to a nearly flat sheet with faint wall
  imprints. Yet every seeded run ends at a HIGHER final cost (1445 vs 1167
  on simu_76_noise), so the residual set genuinely scores the flat map
  better: at weight 0.0349 the eikonal term cannot hold |grad| = 1 against
  scan and hallucination, and flattening is the cheaper way to cut cost.
  Read cost as a solver diagnostic, never as map quality.
- A constant-0 map makes the starting pose gradient exactly zero
  (2026-09-16). At u = 0 the scan residual vanishes for any pose, the
  hallucination residual equals its asserted offset, the eikonal is
  pose-free, and a batch run's odometry residual is zero because the
  initial poses are the odometry. Four simu datasets whose odometry differs
  by up to 2.7 m all report the same initial cost, 12960.2. Poses cannot
  move until the map builds structure. Seeding from scans removes the dead
  start: initial cost then varies by dataset (11148-12915).
- Keep the map signed. Signed values are what make a surface addressable
  below cell size: the interpolant between a positive and a negative node
  crosses zero exactly at the wall, at slope 1, so the scan residual is
  linear in the pose error. Unsigned node values put both flanking nodes
  above zero, so the interpolant never reaches zero and the surface can only
  sit at a node. The registration force then depends on where the wall falls
  inside the cell: slope 0.8 at a tenth of a cell, 0.5 at a quarter, exactly
  0 at the midpoint, and negative beyond it. A wall at a cell midpoint is
  locally unobservable. This is why 0.5 m cells still reach 0.029 m on ACES.
- Seeding the map from scans is a dead end as implemented (2026-09-16,
  `map.init_from_scans`, default off). Results across the full grid:
  batch on simu, GT relative translation for constant-0 / signed seed /
  unsigned seed — noise 0.0069 / 0.0063 / 0.0087, more_noise 0.0922 /
  0.0744 / 0.0910, very_noise 0.1384 / 0.1312 / 0.0762, extreme_noise
  0.1310 / 0.2366 / 0.0826. Incremental on simu — noise 0.0029 / 0.0806 /
  0.0440, very_noise 0.0504 / 0.1725 / 0.0572. Incremental on real data,
  ACES slice against the published relations — 0.022 / 0.043 / 0.047 median;
  Intel lap-1 revisit consistency — 0.011 / 0.375 / 0.411, with the revisit
  frame count falling 100 / 69 / 56, so both seeded runs drifted. Batch on
  real data, ACES slice — 0.024 / 0.023 / 0.025, no difference at all.
  The one cell that improves is batch on the degraded simu rungs, where a
  seed built from the whole dataset pulls the solve out of the double-wall
  minimum. Everywhere else the seed is neutral or harmful. An incremental
  run may honestly seed only from frame 0, and one scan states confident
  wrong values everywhere it did not observe; later scans must then climb
  out of them. A confident partial prior is worse than a neutral one.
- The projective seed sign is wrong in principle, which is why the signed
  seed loses to the unsigned one in most cells. It calls every occluded cell
  inside matter, so the field jumps from a large negative to a large positive
  value where an occluded region meets free space seen from elsewhere, and no
  eikonal solution holds such a seam. A sign is only definable where a
  measurement defines it. That argues for a local sign taken from the surface
  normal, not a global inside-outside rule.
- Seed the map only where scans have been observed. Batch runs seed from
  the whole dataset and gain map quality at roughly unchanged pose accuracy
  (GT relative translation 0.0069 to 0.0063, 0.0922 to 0.0744, 0.1384 to
  0.1312, but 0.1310 to 0.2366 on simu_76_extreme_noise). An incremental
  run may only seed from frame 0, and that loses badly: 0.0029 to 0.0806 on
  simu_76_noise and 0.0504 to 0.1725 on simu_76_very_noise. One scan sees a
  fraction of the scene, so its distance field states confident wrong values
  everywhere it did not observe, and later scans must climb out of them. A
  confident partial prior is worse than a neutral one. `init_from_scans`
  stays off by default; the variant worth testing re-seeds only cells as
  they are first observed.

- Full-Intel runs optimize the dense state: ~42 min at 100x100 instead of ~16 min with the active region.
- The pose Jacobian is deliberately inconsistent with the residual's true derivative; finite-difference Jacobian tests cover the default (exact) mode only.
- DEC-0003's `weighted` normals stay the default for the simulated datasets; the Intel configs override to `pca`.

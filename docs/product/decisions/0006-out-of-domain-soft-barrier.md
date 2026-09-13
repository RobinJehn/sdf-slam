---
id: 0006
title: Out-of-domain points keep a soft barrier residual
type: architecture
owner: Robin Jehn
created: 2026-09-13
updated: 2026-09-13
requirements:
- ../PRD.md
---

# DEC-0006 — Out-of-domain points keep a soft barrier residual

## Decision

A point that falls outside the map domain keeps its residual as `(0 - desired)` with zero Jacobian toward the map: the SDF reads as 0 there and only the pose block receives gradient through the desired value. The residual row is never dropped. This matches the original dissertation implementation.

## Context

The first implementation deleted residual rows for out-of-domain points. Deleting a row rewards escape: a trajectory that pushes scan points past the domain edge removes their cost, so tight map domains caused trajectory runaways (all pre-barrier tight-box Intel runs escaped; see the 2026-09-13 campaign in the README). With the barrier kept, the same configs stay anchored. The soft barrier applies in `EvalPointResidual` (residual math), `Problem::Cost`, and the LM solver's row assembly, so cost, gradient, and reported metrics agree.

## Alternatives considered

- Drop out-of-domain rows (tried, reverted 2026-09-12) — rewards escape, causes runaways under tight domains.
- Hard clamp of poses to the domain (considered) — distorts the optimum and hides the modelling problem.
- Extrapolate the SDF beyond the domain (considered) — the Ceres backend gets this for free from frozen cells (DEC-0004), but full-Intel Ceres diverges, so extrapolation is not a substitute for the barrier.

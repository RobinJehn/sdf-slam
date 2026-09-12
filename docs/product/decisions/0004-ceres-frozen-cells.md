---
id: 0004
title: Ceres backend freezes grid cells per solve
type: architecture
owner: Robin Jehn
created: 2026-09-12
updated: 2026-09-12
requirements:
- ../PRD.md
---

# DEC-0004 — Ceres backend freezes grid cells per solve

## Decision

The Ceres backend freezes each point residual's grid cell at problem-build time: the four node parameter blocks stay fixed for the whole solve, while alpha/beta interpolation weights are recomputed from the current pose in every evaluation. A point that leaves its build-time cell extrapolates that cell's bilinear patch. The hand-rolled LM backend re-derives cells every iteration.

## Context

Ceres parameter blocks per residual cannot change during a solve. Alternatives all cost more than they return: one whole-map parameter block produces dense 1 x N Jacobian rows; rebuilding the Ceres problem every iteration discards the solver's internal state. Frozen cells are exact while the point stays in its cell and a controlled linear extrapolation when it briefly leaves; incremental mode rebuilds the problem every increment anyway, so the freeze window is short.

## Alternatives considered

- Whole map as one parameter block (considered): memory blowup on 10k+ nodes.
- Rebuild the Ceres problem each iteration (considered): correct but slow; equivalent to writing our own outer LM loop around Ceres.

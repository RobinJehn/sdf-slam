---
id: 0003
title: Normal estimation improves via ablation of two methods
type: design
owner: Robin Jehn
created: 2026-09-12
updated: 2026-09-12
requirements:
- ../PRD.md
---

# DEC-0003 — Normal estimation improves via ablation of two methods

## Decision

Ship three grid-node normal methods behind `normals.method` and compare them by ablation on the simulated dataset: `pca` (dissertation baseline: nearest scan point's kNN-PCA normal), `hybrid` (blend the PCA normal with the sign-matched vector toward the nearest scan point, weighted by the PCA eigenvalue ratio), and `weighted` (PCA normal, Eikonal weight scaled down by cornerness).

## Context

The dissertation traces its worst map errors to normal estimation at corners and equidistant regions (sections 4.1.1, 5.2) and proposes the closest-point vector as an extra cue (6.3). `hybrid` implements that cue; `weighted` instead removes the damage a wrong normal does. Which one wins is an empirical question, so both stay until the ablation decides.

## Alternatives considered

- Pick one method up front (considered): no evidence either way; the ablation is cheap because the method is a config switch.

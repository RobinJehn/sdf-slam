---
id: 0003
title: Normal estimation improves via ablation of two methods
type: design
owner: Robin Jehn
created: 2026-09-12
updated: 2026-09-13
requirements:
- ../PRD.md
---

# DEC-0003 — Normal estimation improves via ablation of two methods

## Decision

Grid-node normals default to the `weighted` method: the nearest scan point's kNN-PCA normal, with the Eikonal residual weight scaled down by the neighborhood's cornerness (eigenvalue ratio). `pca` (dissertation baseline) and `hybrid` (blend with the sign-matched closest-point vector) stay available behind `normals.method` for comparison.

Ablation on simu_76 (2026-09-12, 100 LM iterations): on clean odometry the three methods tie. With noise [0.1, 0.1, 0.01], `weighted` cuts the mean relative rotation error to 0.00017 rad versus 0.00049 (`pca`) and 0.00046 (`hybrid`), and the absolute rotation error to 0.017 rad versus 0.030/0.026, at equal translation error.

A re-run under the smooth-gradient default (2026-09-13, same noise config) keeps the ranking: `weighted` wins every metric (mean translation 0.220 versus 0.245 `pca` / 0.283 `hybrid`; mean relative rotation 0.00020 versus 0.00027 / 0.00034 rad). The margin narrows from 3x to about 1.4x on relative rotation. Note the Intel flagship recipe (DEC-0007) still sets `pca` with `k_neighbors: 7` explicitly — that combination is part of the recipe equilibrium and this simu ablation does not override it.

## Context

The dissertation traces its worst map errors to normal estimation at corners and equidistant regions (sections 4.1.1, 5.2) and proposes the closest-point vector as an extra cue (6.3). `hybrid` implements that cue; `weighted` instead removes the damage a wrong normal does. Which one wins is an empirical question, so both stay until the ablation decides.

## Alternatives considered

- `pca` as default (considered): dissertation baseline; loses 3x on relative rotation error under odometry noise.
- `hybrid` closest-point blend (tried, not adopted 2026-09-12): implements the dissertation's 6.3 suggestion but does not beat `pca` on trajectory metrics; a wrong-but-confident normal still hurts, whereas down-weighting removes the damage.

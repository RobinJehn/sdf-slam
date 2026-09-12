---
id: 0002
title: Active-region state vector over the SDF grid
type: design
owner: Robin Jehn
created: 2026-09-12
updated: 2026-09-12
requirements:
- ../PRD.md
---

# DEC-0002 — Active-region state vector over the SDF grid

## Decision

The state vector optionally contains only "active" grid nodes: nodes of cells touched by scan or hallucinated points at the initial poses, dilated by `active_margin` cells. Inactive nodes stay constant; Eikonal residuals exist only where all three referenced nodes are active. Off by default (`active_region: false` reproduces the dissertation's full-grid state).

## Context

The dissertation (section 5.1) flags the statically sized map as wasted computation: on sparse trajectories most of the 100x100 grid is never observed. Restricting the state shrinks the normal equations without changing observed-region behavior. The margin gives poses room to move during a solve.

## Alternatives considered

- Rebuilding the active set every LM iteration (considered): exact, but incompatible with Ceres' fixed parameter blocks and rarely needed at margin >= 2.
- Dynamically growing map allocation (considered): only helps unbounded exploration; domain auto-sizing (`map.auto_domain`) covers the vendored datasets.

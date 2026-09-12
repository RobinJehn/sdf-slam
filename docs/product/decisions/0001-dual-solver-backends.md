---
id: 0001
title: Two solver backends over one shared residual layer
type: architecture
owner: Robin Jehn
created: 2026-09-12
updated: 2026-09-12
requirements:
- ../PRD.md
---

# DEC-0001 — Two solver backends over one shared residual layer

## Decision

Keep two solver backends behind one shared analytic-residual layer (`src/problem/residual_math.hpp`): a hand-rolled Levenberg-Marquardt solver (Eigen sparse Cholesky, dissertation algorithm 1) and a Ceres backend. `bench_solvers` compares them on identical problems.

## Context

The hand-rolled solver reproduces the dissertation exactly and re-derives grid cells every iteration. Ceres brings a robust trust region, multithreaded solves, and a correctness cross-check for the shared Jacobians. The benchmark quantifies the trade-off instead of arguing about it.

## Alternatives considered

- Ceres only (considered): loses the exact dissertation semantics (fixed damping, per-iteration cell re-derivation) and hides the sparsity structure the dissertation analyzes.
- Hand-rolled only (considered): no independent check of the analytic Jacobians beyond finite differences; perf work all manual.

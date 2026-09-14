---
id: product
title: "sdf-slam: unified SDF SLAM, improved reimplementation"
type: product
status: draft
owner: Robin Jehn
created: 2026-09-12
updated: 2026-09-14
supersedes: null
superseded_by: null
decisions:
- decisions/0001-dual-solver-backends.md
- decisions/0002-active-region-map.md
- decisions/0003-normal-estimation-ablation.md
- decisions/0004-ceres-frozen-cells.md
- decisions/0005-lean-dependency-stack.md
- decisions/0006-out-of-domain-soft-barrier.md
- decisions/0007-smooth-gradient-registration.md
- decisions/0008-bitwise-exact-performance-work.md
- decisions/0009-parameter-selection-by-ensembles.md
sources: []
---

# PRD — sdf-slam: unified SDF SLAM, improved reimplementation

## Overview

sdf-slam is an improved C++ reimplementation of the MInf dissertation "SDF SLAM" (Robin Jehn, University of Edinburgh, 2026). The system jointly optimizes 2D sensor poses and a continuous signed distance function (SDF) map in one nonlinear least-squares problem. The original proof-of-concept lives in `../sdf_slam` and stays untouched as a reference.

## Goals

1. Reproduce the dissertation's methodology (chapter 3): scan, hallucination, Eikonal, and odometry residuals over a bilinearly interpolated SDF grid, solved with Levenberg-Marquardt.
2. Improve performance: parallel residual/Jacobian assembly and an active-region map.
3. Improve normal estimation: ablation of a hybrid closest-point blend versus confidence-weighted Eikonal residuals.
4. Keep two solver backends (hand-rolled LM, Ceres) behind one shared residual layer and benchmark them.

## Non-goals (v1)

- Loop closure.
- Sliding-window optimization.
- 3D extension.
- Ground-truth map error metric (needs a scene-to-SDF generator; trajectory metrics cover evaluation for now).

## Functional requirements

- FR-1: `run_slam <config.yaml>` executes a batch or incremental run and writes poses, map, and metrics to an output directory.
- FR-2: Every analytic Jacobian is verified against finite differences in the test suite.
- FR-3: The dissertation's simulated experiments (4.1.1, 4.1.3) run from vendored configs and datasets.
- FR-4: `bench_solvers` compares LM and Ceres backends on the same problem.
- FR-5: Python scripts in `tools/viz` render map and trajectory figures from run outputs.

## Current results

Batch on simu_76 (dissertation experiment 4.1.1, table 4.2 in parentheses): mean translation error 0.039 (0.022), mean relative translation error 0.0041 (0.0031), mean relative rotation error 0.00016 (0.00014). 100 iterations run in ~9 s on an M-series laptop. With noise [0.1, 0.1, 0.01]: relative translation error 0.033 (0.029); the absolute error is 1.11 (0.21) - the drift-toward-fixed-frame effect the dissertation describes in 4.1.3, currently stronger here.

## Constraints

- Lean dependency stack: Eigen + Ceres from the system, nanoflann/yaml-cpp/googletest/benchmark via FetchContent. No PCL, no OpenCV.
- Data is vendored under `data/` (simulated sets + Intel Research Lab).

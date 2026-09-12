---
id: 0005
title: Lean dependency stack with Python visualization
type: tooling
owner: Robin Jehn
created: 2026-09-12
updated: 2026-09-12
requirements:
- ../PRD.md
---

# DEC-0005 — Lean dependency stack with Python visualization

## Decision

The C++ core depends on Eigen and Ceres from the system package manager plus nanoflann, yaml-cpp, googletest, and google benchmark via FetchContent. No PCL, no OpenCV, no Python bindings; figures come from standalone matplotlib scripts in `tools/viz` reading the run output files.

## Context

The original repo pulled PCL only for 2D normal estimation (via a vendored fork) and OpenCV only for display. kNN + 2D PCA is ~60 lines with nanoflann and Eigen; ASCII PCD parsing is trivial. Dropping the heavy dependencies cuts build time from minutes to seconds and makes CI a two-package apt install.

## Alternatives considered

- Keep PCL/OpenCV (considered): known-working but slow builds and a vendored PCL patch to maintain.
- C++ visualization (considered): every option adds a windowing dependency; Python plots are better for the dissertation-style figures anyway.

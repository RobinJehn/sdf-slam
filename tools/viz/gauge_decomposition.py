#!/usr/bin/env python3
"""Decompose the trajectory spread between runs into global and differential parts.

The frame-0 anchor is exact but soft at range (DEC-0007): jittered runs of
one config disagree by meters in absolute pose while relative poses stay
tight. This tool measures that structure for a set of runs, pairwise:

- rigid_residual: raw position deviation vs the deviation left after the
  best single global SE(2) fit. A small remainder means the runs differ by
  a rigid lean; a large remainder means the deviation is differential.
- gap_deviation: deviation of the gap-g relative transforms. This is
  invariant under any global rigid motion, so it isolates the differential
  component at length scale g. Loop closures stiffen it only inside the
  frame range their pairs cover.

Usage: uv run gauge_decomposition.py <run_dir> <run_dir> [...] \
    [--gaps 1,50,200] [--region lo:hi]
see DEC-0007 smooth-gradient-registration
"""

import argparse
import itertools
from pathlib import Path

import numpy as np

from icp_relations import rigid_fit
from plot_map import load_poses


def relative_transforms(poses: np.ndarray, gap: int) -> np.ndarray:
    """Gap-g relative transforms (dx, dy, dtheta), each in frame i's frame."""
    x, y, theta = poses[:, 0], poses[:, 1], poses[:, 2]
    dx, dy = x[gap:] - x[:-gap], y[gap:] - y[:-gap]
    c, s = np.cos(theta[:-gap]), np.sin(theta[:-gap])
    unwrapped = np.unwrap(theta)
    dtheta = unwrapped[gap:] - unwrapped[:-gap]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, dtheta], axis=1)


def gap_deviation(poses_a: np.ndarray, poses_b: np.ndarray, gap: int) -> np.ndarray:
    """Translation deviation between the runs' gap-g relative transforms."""
    rel_a = relative_transforms(poses_a, gap)
    rel_b = relative_transforms(poses_b, gap)
    return np.linalg.norm(rel_a[:, :2] - rel_b[:, :2], axis=1)


def rigid_residual(poses_a: np.ndarray, poses_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame position deviation, raw and after the best global SE(2) fit."""
    pa, pb = poses_a[:, :2], poses_b[:, :2]
    raw = np.linalg.norm(pa - pb, axis=1)
    transform = rigid_fit(pa, pb)
    aligned = pa @ transform[:2, :2].T + transform[:2, 2]
    internal = np.linalg.norm(aligned - pb, axis=1)
    return raw, internal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--gaps", default="1,50,200",
                        help="comma-separated frame gaps for the differential metric")
    parser.add_argument("--region", default=None,
                        help="lo:hi frame slice, e.g. 0:681 for the closure-covered part")
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        parser.error("need at least two run dirs")

    poses = {d.name: load_poses(d / "poses_estimated.csv") for d in args.run_dirs}
    if args.region:
        lo, hi = (int(v) for v in args.region.split(":"))
        poses = {name: p[lo:hi] for name, p in poses.items()}
    gaps = [int(g) for g in args.gaps.split(",")]

    print("== global SE(2) fit ==")
    for a, b in itertools.combinations(poses, 2):
        raw, internal = rigid_residual(poses[a], poses[b])
        print(f"{a} vs {b}: raw mean {raw.mean():.4f} max {raw.max():.4f} | "
              f"internal mean {internal.mean():.4f} max {internal.max():.4f}")

    print("\n== differential (gap) deviation, pooled over pairs ==")
    for gap in gaps:
        devs = np.concatenate([
            gap_deviation(poses[a], poses[b], gap)
            for a, b in itertools.combinations(poses, 2)
        ])
        print(f"gap {gap:4d}: trans dev mean {devs.mean():.4f} "
              f"p95 {np.percentile(devs, 95):.4f} max {devs.max():.4f}")


if __name__ == "__main__":
    main()

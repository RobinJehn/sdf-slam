#!/usr/bin/env python3
"""Replace a dataset's wheel odometry with scan-matched (ICP) odometry.

For every consecutive scan pair the tool refines the odometry relative pose
with point-to-line ICP (seeded by the wheel odometry). Pairs that fail the
confidence guards (residual, inliers, translation observability) keep the
wheel value. The refined relative poses integrate to absolute poses, written
as a new dataset directory with symlinked scans and a new scanner_info.txt —
the solver's odometry residuals then constrain scan-matched motion instead
of wheel motion, with no solver change.

Usage: uv run icp_odometry.py --dataset <dir> --out <dir> [--max-residual 0.1]

see DEC-0009 parameter-selection-by-ensembles
"""

import argparse
from pathlib import Path

import numpy as np

from icp_relations import (
    icp,
    matrix_pose,
    pose_matrix,
    relative_pose,
    target_normals,
    translation_observability,
)
from make_video import load_scan


def refine_odometry(
    scans: list[np.ndarray],
    poses: np.ndarray,
    max_residual: float = 0.1,
    min_inliers: int = 50,
    min_observability: float = 0.15,
) -> tuple[np.ndarray, int]:
    """ICP-refined absolute poses and the count of refined pairs.

    Pair (i-1, i) keeps the wheel relative pose when ICP fails a guard.
    """
    absolute = [np.asarray(poses[0], dtype=float)]
    refined = 0
    for i in range(1, len(scans)):
        seed = relative_pose(poses[i - 1], poses[i])
        pose, residual, inliers = icp(scans[i], scans[i - 1], seed)
        good = residual <= max_residual and inliers >= min_inliers
        if good:
            observability = translation_observability(target_normals(scans[i - 1]))
            good = observability >= min_observability
        rel = pose if good else seed
        refined += int(good)
        absolute.append(matrix_pose(pose_matrix(absolute[-1]) @ pose_matrix(rel)))
    return np.array(absolute), refined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-residual", type=float, default=0.1)
    parser.add_argument("--min-inliers", type=int, default=50)
    parser.add_argument("--min-observability", type=float, default=0.15)
    args = parser.parse_args()

    scan_paths = sorted(args.dataset.glob("scan*.pcd"))
    scans = [load_scan(p) for p in scan_paths]
    poses = np.loadtxt(args.dataset / "scanner_info.txt", ndmin=2)

    absolute, refined = refine_odometry(
        scans, poses, args.max_residual, args.min_inliers, args.min_observability
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for p in scan_paths:
        link = args.out / p.name
        if not link.exists():
            link.symlink_to(p.resolve())
    np.savetxt(args.out / "scanner_info.txt", absolute, fmt="%.9f")
    print(f"refined {refined}/{len(scans) - 1} pairs -> {args.out}")


if __name__ == "__main__":
    main()

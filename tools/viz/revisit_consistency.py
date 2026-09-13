#!/usr/bin/env python3
"""Score the global consistency of a run by loop-closure agreement.

For every pair of frames that observe the same area but lie far apart in
time, the scan points of the later frame must land on the walls the earlier
frame already drew. The score is the median nearest-neighbor distance from
each frame's points to the points of all frames at least `--min-gap` frames
older, restricted to frame pairs whose poses are within `--radius` meters.
Lower is better; drift ghosts raise it sharply while local-fit metrics miss
them.

Usage: uv run revisit_consistency.py <run_dir> [<run_dir> ...] --dataset <dir>
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from make_video import load_scan, transform
from plot_map import load_poses


def revisit_pairs(poses: np.ndarray, min_gap: int, radius: float) -> dict[int, list[int]]:
    """Frame -> list of frames at least min_gap older with a nearby pose."""
    pairs = {}
    n = len(poses)
    for j in range(min_gap, n):
        old_ids = [i for i in range(0, j - min_gap)
                   if np.hypot(poses[i, 0] - poses[j, 0], poses[i, 1] - poses[j, 1]) < radius]
        if old_ids:
            pairs[j] = old_ids
    return pairs


def score_run(run_dir: Path, scans: list[np.ndarray], pairs: dict[int, list[int]]):
    poses = load_poses(run_dir / "poses_estimated.csv")
    clouds = [transform(scans[i], poses[i]) for i in range(len(poses))]

    distances = []
    for j, old_ids in pairs.items():
        reference = np.vstack([clouds[i] for i in old_ids])
        tree = cKDTree(reference)
        d, _ = tree.query(clouds[j], k=1)
        distances.append(d)
    if not distances:
        return None
    return np.concatenate(distances)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--min-gap", type=int, default=200, help="minimum frame separation")
    parser.add_argument("--radius", type=float, default=3.0, help="pose distance for a revisit")
    parser.add_argument(
        "--pairs-from",
        type=Path,
        default=None,
        help="run dir whose estimate defines the revisit pairs for every run. "
        "Default: each run scores on its own pairs (internal consistency). "
        "There is no ground truth, so neither mode is bias-free: own pairs "
        "favor compressed estimates, shared pairs favor the reference run. "
        "Always confirm the ranking against the rendered maps.",
    )
    args = parser.parse_args()

    scans = [load_scan(p) for p in sorted(args.dataset.glob("scan*.pcd"))]
    shared_pairs = None
    if args.pairs_from is not None:
        shared_pairs = revisit_pairs(load_poses(args.pairs_from / "poses_estimated.csv"),
                                     args.min_gap, args.radius)
        print(f"revisit pairs from {args.pairs_from.name}: {len(shared_pairs)} frames")

    print(f"{'run':32s} {'median':>8s} {'p90':>8s} {'frames':>7s}")
    for run_dir in args.run_dirs:
        pairs = shared_pairs
        if pairs is None:
            pairs = revisit_pairs(load_poses(run_dir / "poses_estimated.csv"),
                                  args.min_gap, args.radius)
        d = score_run(run_dir, scans, pairs)
        if d is None:
            print(f"{run_dir.name:32s} {'—':>8s} {'—':>8s} {0:7d}")
            continue
        print(f"{run_dir.name:32s} {np.median(d):8.3f} {np.percentile(d, 90):8.3f} {len(pairs):7d}")


if __name__ == "__main__":
    main()

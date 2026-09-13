#!/usr/bin/env python3
"""Animate an incremental run from its snapshots: each frame shows the map
after one more increment, the re-optimized trajectory, all scan points so far
(black), and the newest scan (red).

Requires a run with `snapshot_every` > 0 and ffmpeg on the PATH.

Usage: uv run make_video.py <run_output_dir> --dataset <dataset_dir> [--save run.mp4]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from plot_map import load_map, load_poses, upsample_bilinear


def load_scan(path: Path) -> np.ndarray:
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("DATA")) + 1
    return np.loadtxt(lines[start:], ndmin=2)


def transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])
    return points @ rotation.T + pose[:2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset dir with scan*.pcd")
    parser.add_argument("--save", type=Path, default=None, help="output mp4 (default: run_dir/run.mp4)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--every", type=int, default=1, help="use every Nth snapshot")
    parser.add_argument("--vlim", type=float, default=3.0, help="color scale limit")
    parser.add_argument(
        "--upsample",
        type=lambda s: max(int(s), 1),
        default=4,
        help="bilinear samples per grid cell (minimum 1)",
    )
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--pad", type=float, default=2.0, help="padding around the final scan bbox")
    args = parser.parse_args()

    snapshot_dir = args.run_dir / "snapshots"
    map_paths = sorted(snapshot_dir.glob("map_*.txt"))
    if not map_paths:
        raise SystemExit(f"no snapshots in {snapshot_dir}; rerun with snapshot_every > 0")
    map_paths = map_paths[:: args.every] + ([map_paths[-1]] if (len(map_paths) - 1) % args.every else [])

    scans = [load_scan(p) for p in sorted(args.dataset.glob("scan*.pcd"))]

    # The view stays fixed at the final scan footprint so the camera does not
    # jump while the trajectory is re-optimized.
    final_tag = re.search(r"map_(\d+)", map_paths[-1].stem).group(1)
    final_poses = load_poses(snapshot_dir / f"poses_{final_tag}.csv")
    final_points = np.vstack(
        [transform(scans[i], final_poses[i]) for i in range(len(final_poses))]
    )
    x_min, y_min = final_points.min(axis=0) - args.pad
    x_max, y_max = final_points.max(axis=0) + args.pad

    fig, ax = plt.subplots(figsize=(9, 8))
    values, extent = load_map(map_paths[0])
    image = ax.imshow(
        upsample_bilinear(values, args.upsample),
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=-args.vlim,
        vmax=args.vlim,
    )
    past = ax.scatter([], [], s=0.5, c="black", marker=".", linewidths=0, label="scan points")
    current = ax.scatter([], [], s=2.0, c="red", marker=".", linewidths=0, label="newest scan")
    (path_line,) = ax.plot([], [], "-", color="tab:green", linewidth=1.2, label="estimated path")
    ax.set_xlim(max(x_min, extent[0]), min(x_max, extent[1]))
    ax.set_ylim(max(y_min, extent[2]), min(y_max, extent[3]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    legend = ax.legend(loc="upper right")
    for handle in legend.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([30.0])
    fig.colorbar(image, ax=ax, label="signed distance")

    save_path = args.save or args.run_dir / "run.mp4"
    writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, save_path, dpi=args.dpi):
        for map_path in map_paths:
            tag = re.search(r"map_(\d+)", map_path.stem).group(1)
            poses = load_poses(snapshot_dir / f"poses_{tag}.csv")
            n = len(poses)

            values, _ = load_map(map_path)
            image.set_data(upsample_bilinear(values, args.upsample))
            past.set_offsets(
                np.vstack([transform(scans[i], poses[i]) for i in range(n - 1)])
                if n > 1
                else np.empty((0, 2))
            )
            current.set_offsets(transform(scans[n - 1], poses[n - 1]))
            path_line.set_data(poses[:, 0], poses[:, 1])
            ax.set_title(f"Incremental SDF SLAM — {n} of {len(scans)} scans")
            writer.grab_frame()
    print(f"saved {save_path} ({len(map_paths)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()

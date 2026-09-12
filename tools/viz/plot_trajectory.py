#!/usr/bin/env python3
"""Plot estimated vs odometry (and ground truth, when present) trajectories.

Usage: uv run plot_trajectory.py <run_output_dir> [--save fig.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_poses(path: Path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 7))

    ground_truth = args.run_dir / "poses_ground_truth.csv"
    if ground_truth.exists():
        gt = load_poses(ground_truth)
        ax.plot(gt[:, 0], gt[:, 1], ".-", color="tab:blue", markersize=3, label="ground truth")

    odometry = load_poses(args.run_dir / "poses_odometry.csv")
    ax.plot(odometry[:, 0], odometry[:, 1], ".--", color="tab:green", markersize=3,
            label="odometry")

    estimated = load_poses(args.run_dir / "poses_estimated.csv")
    ax.plot(estimated[:, 0], estimated[:, 1], ".-", color="tab:red", markersize=3,
            label="estimated")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory comparison")
    ax.axis("equal")
    ax.legend()

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

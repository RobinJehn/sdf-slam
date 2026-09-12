#!/usr/bin/env python3
"""Plot the reconstructed SDF map of a run with the estimated path overlaid.

Usage: uv run plot_map.py <run_output_dir> [--save fig.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_map(path: Path):
    with open(path) as f:
        header = f.readline().split()
    nx, ny = int(header[0]), int(header[1])
    x0, y0, dx, dy = map(float, header[2:6])
    values = np.loadtxt(path, skiprows=1)
    assert values.shape == (ny, nx), f"map shape {values.shape} != ({ny}, {nx})"
    extent = (x0, x0 + (nx - 1) * dx, y0, y0 + (ny - 1) * dy)
    return values, extent


def load_poses(path: Path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--vlim", type=float, default=10.0, help="color scale limit")
    args = parser.parse_args()

    values, extent = load_map(args.run_dir / "map.txt")
    poses = load_poses(args.run_dir / "poses_estimated.csv")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(
        values,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=-args.vlim,
        vmax=args.vlim,
    )
    ax.plot(poses[:, 0], poses[:, 1], ".-", color="tab:green", markersize=3, label="estimated path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Estimated SDF map")
    ax.legend()
    fig.colorbar(image, ax=ax, label="signed distance")

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

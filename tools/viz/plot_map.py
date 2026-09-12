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
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="crop the view to the region where the map deviates from its initial value",
    )
    parser.add_argument("--zoom-pad", type=float, default=2.0, help="padding around the zoom box")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    values, extent = load_map(args.run_dir / "map.txt")
    poses = load_poses(args.run_dir / "poses_estimated.csv")

    fig, ax = plt.subplots(figsize=(10, 9))
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

    if args.zoom:
        # Optimized nodes deviate from the (constant) initial map value; crop
        # to their bounding box plus the trajectory.
        background = np.bincount(
            np.digitize(values.ravel(), np.unique(values.ravel())) - 1
        ).argmax()
        modal_value = np.unique(values.ravel())[background]
        rows, cols = np.nonzero(values != modal_value)
        if rows.size > 0:
            x0, x1, y0, y1 = extent
            nx = values.shape[1]
            ny = values.shape[0]
            dx = (x1 - x0) / (nx - 1)
            dy = (y1 - y0) / (ny - 1)
            xs = np.concatenate([x0 + cols * dx, poses[:, 0]])
            ys = np.concatenate([y0 + rows * dy, poses[:, 1]])
            ax.set_xlim(xs.min() - args.zoom_pad, xs.max() + args.zoom_pad)
            ax.set_ylim(ys.min() - args.zoom_pad, ys.max() + args.zoom_pad)

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

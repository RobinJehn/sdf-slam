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


def upsample_bilinear(values: np.ndarray, factor: int) -> np.ndarray:
    """Evaluate the map's bilinear surface (dissertation eq. 3.7) on a raster
    with `factor` samples per cell. The map model is piecewise bilinear between
    nodes, so this renders the true model surface; accuracy stays capped by the
    node spacing."""
    ny, nx = values.shape
    fx = np.linspace(0.0, nx - 1, (nx - 1) * factor + 1)
    fy = np.linspace(0.0, ny - 1, (ny - 1) * factor + 1)
    w = np.clip(fx.astype(int), 0, nx - 2)
    h = np.clip(fy.astype(int), 0, ny - 2)
    alpha = (fx - w)[None, :]
    beta = (fy - h)[:, None]
    v00 = values[np.ix_(h, w)]
    v10 = values[np.ix_(h, w + 1)]
    v01 = values[np.ix_(h + 1, w)]
    v11 = values[np.ix_(h + 1, w + 1)]
    return (
        v00 * (1 - alpha) * (1 - beta)
        + v10 * alpha * (1 - beta)
        + v01 * (1 - alpha) * beta
        + v11 * alpha * beta
    )


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
    parser.add_argument(
        "--upsample",
        type=int,
        default=8,
        help="bilinear samples per grid cell; 1 shows raw node values",
    )
    parser.add_argument(
        "--no-scans",
        action="store_true",
        help="do not overlay scan points even when scan_points_global.csv exists",
    )
    args = parser.parse_args()

    values, extent = load_map(args.run_dir / "map.txt")
    poses = load_poses(args.run_dir / "poses_estimated.csv")
    rendered = upsample_bilinear(values, args.upsample) if args.upsample > 1 else values

    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(
        rendered,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=-args.vlim,
        vmax=args.vlim,
    )
    ax.plot(poses[:, 0], poses[:, 1], ".-", color="tab:green", markersize=3, label="estimated path")

    scan_points_path = args.run_dir / "scan_points_global.csv"
    if scan_points_path.exists() and not args.no_scans:
        scan_points = np.loadtxt(scan_points_path, delimiter=",", skiprows=1)
        ax.scatter(
            scan_points[:, 0],
            scan_points[:, 1],
            s=0.5,
            c="black",
            marker=".",
            linewidths=0,
            rasterized=True,
            label="scan points",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Estimated SDF map")
    legend = ax.legend()
    for handle in legend.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([30.0])
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
            # Clamp to the map extent so the crop never shows blank canvas
            # beyond the domain.
            ax.set_xlim(max(xs.min() - args.zoom_pad, x0), min(xs.max() + args.zoom_pad, x1))
            ax.set_ylim(max(ys.min() - args.zoom_pad, y0), min(ys.max() + args.zoom_pad, y1))

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

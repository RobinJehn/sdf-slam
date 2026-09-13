#!/usr/bin/env python3
"""Measure the stability of a config, not just its score.

The recipe equilibrium is chaotically sensitive (DEC-0007/DEC-0008): a 1e-9
relative change in `lambda_init` can collapse the result. A single run is
therefore a knife-edge sample. This tool runs the same config N times with a
tiny deterministic jitter on `lambda_init`, scores every run with the
revisit-consistency metric, and reports the ensemble distribution. A config
is only as good as its ensemble median, and the spread measures the basin
width — the quantity that predicts whether the config survives a new
environment, compiler, or BLAS.

Usage: uv run stability_ensemble.py --config <yaml> --dataset <dir> [--runs 8]
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from make_video import load_scan
from plot_map import load_poses
from revisit_consistency import revisit_pairs, score_run

LAMBDA_PATTERN = re.compile(r"^(\s*lambda_init:\s*)([0-9.eE+-]+)\s*$", re.MULTILINE)


def jitter_config(text: str, output_dir: str, relative_jitter: float) -> str:
    """Rewrite output_dir and apply a relative jitter to lambda_init."""
    match = LAMBDA_PATTERN.search(text)
    if match is None:
        raise ValueError("config has no lambda_init; nothing to jitter")
    value = float(match.group(2)) * (1.0 + relative_jitter)
    text = LAMBDA_PATTERN.sub(rf"\g<1>{value!r}", text)
    return re.sub(r"^output_dir:.*$", f"output_dir: {output_dir}", text, flags=re.MULTILINE)


def summarize(medians: list[float], divergence_factor: float) -> dict:
    """Ensemble statistics; a run counts as diverged when its median exceeds
    the ensemble best by `divergence_factor`."""
    values = np.array(medians)
    best = values.min()
    return {
        "best": float(best),
        "median": float(np.median(values)),
        "worst": float(values.max()),
        "spread": float(values.max() - best),
        "diverged": int((values > divergence_factor * best).sum()),
        "n": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--jitter", type=float, default=1e-9,
                        help="relative lambda_init jitter per run (run i gets i*jitter)")
    parser.add_argument("--min-gap", type=int, default=200)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--binary", type=Path, default=Path("build/run_slam"))
    parser.add_argument("--divergence-factor", type=float, default=5.0)
    args = parser.parse_args()

    base = args.config.read_text()
    scans = [load_scan(p) for p in sorted(args.dataset.glob("scan*.pcd"))]

    medians = []
    with tempfile.TemporaryDirectory(prefix="stability_") as tmp:
        for i in range(args.runs):
            run_dir = Path(tmp) / f"run_{i}"
            config_path = Path(tmp) / f"run_{i}.yaml"
            config_path.write_text(jitter_config(base, str(run_dir), i * args.jitter))
            subprocess.run([args.binary, config_path], check=True, capture_output=True)
            poses = load_poses(run_dir / "poses_estimated.csv")
            pairs = revisit_pairs(poses, args.min_gap, args.radius)
            distances = score_run(run_dir, scans, pairs)
            median = float(np.median(distances)) if distances is not None else float("inf")
            medians.append(median)
            print(f"run {i} (jitter {i * args.jitter:.1e}): median {median:.3f}", flush=True)

    stats = summarize(medians, args.divergence_factor)
    print(f"\nensemble: median {stats['median']:.3f}, best {stats['best']:.3f}, "
          f"worst {stats['worst']:.3f}, spread {stats['spread']:.3f}, "
          f"diverged {stats['diverged']}/{stats['n']}")


if __name__ == "__main__":
    main()

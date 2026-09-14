#!/usr/bin/env python3
"""Score runs against ICP-verified relative poses (Kümmerle-style relations).

The nearest-neighbor revisit metric rewards compressed estimates. This tool
removes that bias. `build` freezes a revisit pair set from one reference run,
aligns the raw sensor-frame scans of each pair with 2D point-to-point ICP,
and stores the resulting relative poses as the reference relations. `score`
compares the relative poses of any run against these relations and reports
translational and rotational errors. The pair set and the reference poses
never change between runs, so all runs score on identical evidence.

Build:  uv run icp_relations.py build --run <ref_run> --dataset <dir> --out <csv>
Score:  uv run icp_relations.py score --relations <csv> <run_dir> [<run_dir> ...]

see DEC-0009 parameter-selection-by-ensembles
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from make_video import load_scan
from plot_map import load_poses


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    """Homogeneous 3x3 transform for a (x, y, theta) pose."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([[c, -s, pose[0]], [s, c, pose[1]], [0.0, 0.0, 1.0]])


def matrix_pose(transform: np.ndarray) -> np.ndarray:
    """(x, y, theta) pose for a homogeneous 3x3 transform."""
    return np.array(
        [transform[0, 2], transform[1, 2], np.arctan2(transform[1, 0], transform[0, 0])]
    )


def relative_pose(pose_i: np.ndarray, pose_j: np.ndarray) -> np.ndarray:
    """Pose of frame j expressed in frame i."""
    return matrix_pose(np.linalg.inv(pose_matrix(pose_i)) @ pose_matrix(pose_j))


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform that maps source points onto target points.

    Point correspondence is by index. Returns a homogeneous 3x3 transform
    (2D Umeyama without scale).
    """
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    cov = (target - dst_mean).T @ (source - src_mean)
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, d]) @ vt
    translation = dst_mean - rotation @ src_mean
    transform = np.eye(3)
    transform[:2, :2] = rotation
    transform[:2, 2] = translation
    return transform


def target_normals(target: np.ndarray, k_neighbors: int = 5) -> np.ndarray:
    """Unit normal per target point from a PCA line fit over its neighbors."""
    tree = cKDTree(target)
    k = min(k_neighbors, len(target))
    _, indices = tree.query(target, k=k)
    normals = np.zeros_like(target)
    for i, ids in enumerate(np.atleast_2d(indices)):
        neighbors = target[ids]
        centered = neighbors - neighbors.mean(axis=0)
        _, eigenvectors = np.linalg.eigh(centered.T @ centered)
        normals[i] = eigenvectors[:, 0]
    return normals


def icp(
    source: np.ndarray,
    target: np.ndarray,
    init_pose: np.ndarray,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
    max_correspondence: float = 1.0,
) -> tuple[np.ndarray, float, int]:
    """Align source points to target points with point-to-line ICP.

    Point-to-line reaches sub-sample accuracy on wall-like lidar scans, where
    point-to-point stalls on the sampling grid. `init_pose` seeds the
    transform as a (x, y, theta) pose of the source frame in the target
    frame. Correspondences farther than `max_correspondence` after the
    current transform are outliers and do not enter the fit. Iteration stops
    when the incremental step is smaller than `tolerance` (translation in m
    plus rotation in rad). Returns the refined pose, the median perpendicular
    distance of the final inlier correspondences, and the inlier count.
    """
    tree = cKDTree(target)
    normals = target_normals(target)
    transform = pose_matrix(init_pose)
    residual = np.inf
    inliers = 0
    for _ in range(max_iterations):
        moved = source @ transform[:2, :2].T + transform[:2, 2]
        distances, indices = tree.query(moved, k=1)
        keep = distances < max_correspondence
        inliers = int(keep.sum())
        if inliers < 3:
            return matrix_pose(transform), np.inf, inliers
        p = moved[keep]
        q = target[indices[keep]]
        n = normals[indices[keep]]
        # Linearized point-to-line least squares in (dx, dy, dtheta):
        # residual = n . (p + [dx, dy] + dtheta * [-p_y, p_x] - q)
        a = np.column_stack([n[:, 0], n[:, 1], n[:, 0] * -p[:, 1] + n[:, 1] * p[:, 0]])
        b = np.sum(n * (q - p), axis=1)
        step_pose, *_ = np.linalg.lstsq(a, b, rcond=None)
        transform = pose_matrix(step_pose) @ transform
        residual = float(np.median(np.abs(b)))
        if np.hypot(step_pose[0], step_pose[1]) + abs(step_pose[2]) < tolerance:
            break
    return matrix_pose(transform), residual, inliers


def translation_observability(normals: np.ndarray) -> float:
    """Eigenvalue ratio (min/max) of the normal-direction scatter matrix.

    A corridor scan has all normals near one direction: the ratio is near 0
    and point-to-line ICP can slide along the corridor without cost. Scans
    that constrain both translation directions score near 1.
    """
    scatter = normals.T @ normals
    eigenvalues = np.linalg.eigvalsh(scatter)
    return float(eigenvalues[0] / eigenvalues[-1])


def select_pairs(
    poses: np.ndarray, min_gap: int, radius: float, per_frame: int, max_gap: int | None = None
) -> list[tuple[int, int]]:
    """Select (i, j) pairs: for each frame j, the `per_frame` old frames with
    gap j - i in [min_gap, max_gap] (inclusive) and the smallest pose distance
    below `radius`. `max_gap` None leaves the band open above (revisit
    selection); a finite band selects short/medium-gap relation webs."""
    selected = []
    for j in range(len(poses)):

        def distance(i: int, j: int = j) -> float:
            return float(np.hypot(poses[i, 0] - poses[j, 0], poses[i, 1] - poses[j, 1]))

        first = 0 if max_gap is None else max(0, j - max_gap)
        candidates = [i for i in range(first, j - min_gap + 1) if distance(i) < radius]
        candidates.sort(key=distance)
        selected.extend((i, j) for i in candidates[:per_frame])
    return selected


def pose_error(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    """Translational and rotational error between two relative poses."""
    delta = np.linalg.inv(pose_matrix(reference)) @ pose_matrix(estimate)
    pose = matrix_pose(delta)
    return float(np.hypot(pose[0], pose[1])), float(abs(pose[2]))


def build(args: argparse.Namespace) -> None:
    poses = load_poses(args.run / "poses_estimated.csv")
    scans = [load_scan(p) for p in sorted(args.dataset.glob("scan*.pcd"))]
    pairs = select_pairs(poses, args.min_gap, args.radius, args.per_frame, args.max_gap)
    print(f"{len(pairs)} candidate pairs from {args.run.name}")

    rows = []
    dropped = {"residual": 0, "inliers": 0, "observability": 0, "seed_deviation": 0}
    for i, j in pairs:
        seed = relative_pose(poses[i], poses[j])
        pose, residual, inliers = icp(scans[j], scans[i], seed)
        if residual > args.max_residual:
            dropped["residual"] += 1
            continue
        if inliers < args.min_inliers:
            dropped["inliers"] += 1
            continue
        # Sliding and flips keep the residual low; two extra guards catch them.
        target = scans[i]
        tree = cKDTree(target)
        moved = scans[j] @ pose_matrix(pose)[:2, :2].T + pose_matrix(pose)[:2, 2]
        distances, indices = tree.query(moved, k=1)
        matched_normals = target_normals(target)[indices[distances < 1.0]]
        if translation_observability(matched_normals) < args.min_observability:
            dropped["observability"] += 1
            continue
        t_dev, r_dev = pose_error(seed, pose)
        if t_dev > args.max_seed_shift or np.degrees(r_dev) > args.max_seed_rotation:
            dropped["seed_deviation"] += 1
            continue
        rows.append((i, j, pose[0], pose[1], pose[2], residual, inliers))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("i,j,dx,dy,dtheta,residual,inliers\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.9f},{row[3]:.9f},{row[4]:.9f},"
                    f"{row[5]:.4f},{row[6]}\n")
    print(f"kept {len(rows)}, dropped {dropped} -> {args.out}")


def load_relations(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def score(args: argparse.Namespace) -> None:
    relations = load_relations(args.relations)
    print(f"{len(relations)} relations from {args.relations}")
    print(f"{'run':32s} {'t_med':>7s} {'t_mean':>7s} {'r_med°':>7s} {'r_mean°':>8s}")
    for run_dir in args.run_dirs:
        poses = load_poses(run_dir / "poses_estimated.csv")
        trans, rot = [], []
        for row in relations:
            i, j = int(row[0]), int(row[1])
            t_err, r_err = pose_error(row[2:5], relative_pose(poses[i], poses[j]))
            trans.append(t_err)
            rot.append(r_err)
        trans, rot = np.array(trans), np.degrees(rot)
        print(f"{run_dir.name:32s} {np.median(trans):7.3f} {np.mean(trans):7.3f} "
              f"{np.median(rot):7.2f} {np.mean(rot):8.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="freeze pairs and ICP reference relations")
    p_build.add_argument("--run", type=Path, required=True,
                         help="run dir whose estimate defines pairs and ICP seeds")
    p_build.add_argument("--dataset", type=Path, required=True)
    p_build.add_argument("--out", type=Path, required=True)
    p_build.add_argument("--min-gap", type=int, default=200)
    p_build.add_argument("--max-gap", type=int, default=None,
                         help="largest frame gap (inclusive); selects a gap band "
                         "for short/medium relation webs instead of revisits")
    p_build.add_argument("--radius", type=float, default=3.0)
    p_build.add_argument("--per-frame", type=int, default=1,
                         help="reference pairs per revisit frame")
    p_build.add_argument("--max-residual", type=float, default=0.15,
                         help="drop pairs whose ICP median residual exceeds this (m)")
    p_build.add_argument("--min-inliers", type=int, default=50)
    p_build.add_argument("--min-observability", type=float, default=0.15,
                         help="drop pairs whose matched normals leave a translation "
                         "direction unconstrained (corridor sliding)")
    p_build.add_argument("--max-seed-shift", type=float, default=0.7,
                         help="drop pairs where ICP moves this far (m) from the seed; "
                         "such moves are basin escapes, not refinements")
    p_build.add_argument("--max-seed-rotation", type=float, default=20.0,
                         help="drop pairs where ICP rotates this far (deg) from the seed")
    p_build.set_defaults(func=build)

    p_score = sub.add_parser("score", help="score runs against stored relations")
    p_score.add_argument("--relations", type=Path, required=True)
    p_score.add_argument("run_dirs", type=Path, nargs="+")
    p_score.set_defaults(func=score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

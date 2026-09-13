"""Unit tests for icp_relations: known-transform synthetic pairs."""

import numpy as np
import pytest

from icp_relations import (
    icp,
    matrix_pose,
    pose_error,
    pose_matrix,
    relative_pose,
    rigid_fit,
    select_pairs,
    target_normals,
    translation_observability,
)


def l_shaped_scan(n: int = 80) -> np.ndarray:
    """Two perpendicular walls; the corner pins both translation and rotation."""
    along = np.linspace(0.0, 8.0, n)
    wall_x = np.column_stack([along, np.zeros(n)])
    wall_y = np.column_stack([np.zeros(n), along])
    return np.vstack([wall_x, wall_y])


def apply_pose(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    transform = pose_matrix(pose)
    return points @ transform[:2, :2].T + transform[:2, 2]


def test_pose_matrix_roundtrip():
    pose = np.array([1.5, -2.0, 0.7])
    np.testing.assert_allclose(matrix_pose(pose_matrix(pose)), pose, atol=1e-12)


def test_relative_pose_composition():
    pose_i = np.array([2.0, 1.0, 0.3])
    pose_j = np.array([3.5, -0.5, -1.1])
    rel = relative_pose(pose_i, pose_j)
    recomposed = pose_matrix(pose_i) @ pose_matrix(rel)
    np.testing.assert_allclose(matrix_pose(recomposed), pose_j, atol=1e-12)


def test_rigid_fit_recovers_known_transform():
    rng = np.random.default_rng(0)
    source = rng.uniform(-5, 5, size=(40, 2))
    true = np.array([0.8, -1.2, 0.5])
    target = apply_pose(source, true)
    fitted = matrix_pose(rigid_fit(source, target))
    np.testing.assert_allclose(fitted, true, atol=1e-10)


def test_icp_recovers_known_transform_from_offset_seed():
    source = l_shaped_scan()
    true = np.array([0.4, -0.3, 0.15])
    target = apply_pose(source, true)
    seed = true + np.array([0.2, -0.15, 0.05])
    pose, residual, inliers = icp(source, target, seed)
    np.testing.assert_allclose(pose, true, atol=1e-6)
    assert residual < 1e-6
    assert inliers == len(source)


def test_icp_reports_failure_when_no_overlap():
    source = l_shaped_scan()
    target = source + np.array([100.0, 100.0])
    pose, residual, inliers = icp(source, target, np.zeros(3))
    assert residual == np.inf
    assert inliers < 3


def test_pose_error_zero_for_identical_poses():
    pose = np.array([1.0, 2.0, 0.5])
    t_err, r_err = pose_error(pose, pose)
    assert t_err == pytest.approx(0.0, abs=1e-12)
    assert r_err == pytest.approx(0.0, abs=1e-12)


def test_pose_error_wraps_angle():
    reference = np.array([0.0, 0.0, np.pi - 0.05])
    estimate = np.array([0.0, 0.0, -np.pi + 0.05])
    _, r_err = pose_error(reference, estimate)
    assert r_err == pytest.approx(0.1, abs=1e-9)


def test_observability_low_for_corridor_high_for_corner():
    along = np.linspace(0.0, 8.0, 80)
    corridor = np.vstack(
        [np.column_stack([along, np.zeros(80)]), np.column_stack([along, np.full(80, 2.0)])]
    )
    assert translation_observability(target_normals(corridor)) < 0.05
    assert translation_observability(target_normals(l_shaped_scan())) > 0.5


def test_select_pairs_respects_gap_and_radius():
    poses = np.zeros((12, 3))
    poses[:, 0] = np.arange(12) * 10.0
    poses[10] = [0.5, 0.0, 0.0]
    poses[11] = [55.0, 0.0, 0.0]
    pairs = select_pairs(poses, min_gap=5, radius=3.0, per_frame=1)
    assert pairs == [(0, 10)]


def test_select_pairs_picks_nearest_old_frame():
    poses = np.zeros((8, 3))
    poses[0] = [2.0, 0.0, 0.0]
    poses[1] = [0.4, 0.0, 0.0]
    poses[2:7, 0] = np.arange(5) * 20.0 + 100.0
    poses[7] = [0.0, 0.0, 0.0]
    pairs = select_pairs(poses, min_gap=5, radius=3.0, per_frame=1)
    assert pairs == [(1, 7)]

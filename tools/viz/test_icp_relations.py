"""Unit tests for icp_relations: known-transform synthetic pairs."""

import numpy as np
import pytest

from icp_relations import (
    icp,
    matrix_pose,
    pose_error,
    pose_matrix,
    relative_pose,
    residual_weights,
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


def test_select_pairs_gap_band_pairs_every_frame_at_fixed_gap():
    poses = np.zeros((8, 3))
    poses[:, 0] = np.arange(8) * 0.5
    pairs = select_pairs(poses, min_gap=2, radius=1.2, per_frame=1, max_gap=2)
    assert pairs == [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]


def test_select_pairs_max_gap_excludes_older_frames():
    poses = np.zeros((10, 3))
    pairs = select_pairs(poses, min_gap=3, radius=1.0, per_frame=10, max_gap=4)
    assert pairs
    assert all(3 <= j - i <= 4 for i, j in pairs)
    assert (5, 9) in pairs and (6, 9) in pairs


def test_residual_weights_scale_inverse_squared_and_clamp():
    residuals = np.array([0.05, 0.10, 0.20, 0.001, 10.0])
    weights = residual_weights(residuals, cap=4.0)
    assert weights[1] == pytest.approx(1.0)
    assert weights[0] == pytest.approx(4.0)
    assert weights[2] == pytest.approx(0.25)
    assert weights[3] == pytest.approx(4.0)
    assert weights[4] == pytest.approx(0.25)


def test_residual_weights_median_pair_gets_weight_one():
    residuals = np.array([0.04, 0.08, 0.16])
    weights = residual_weights(residuals, cap=100.0)
    assert weights[1] == pytest.approx(1.0)
    assert weights[0] == pytest.approx(4.0)
    assert weights[2] == pytest.approx(0.25)


def test_select_pairs_band_respects_per_frame_and_radius():
    poses = np.zeros((7, 3))
    poses[:, 0] = np.arange(7) * 1.0
    poses[2] = [100.0, 0.0, 0.0]
    pairs = select_pairs(poses, min_gap=1, radius=1.5, per_frame=1, max_gap=3)
    assert (2, 3) not in pairs and (1, 2) not in pairs
    assert (3, 4) in pairs and (0, 1) in pairs
    assert all(pairs.count((i, j)) == 1 for i, j in pairs)

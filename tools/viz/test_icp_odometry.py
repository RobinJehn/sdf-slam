"""Unit tests for icp_odometry: known-motion refinement and guard fallback."""

import numpy as np

from icp_odometry import refine_odometry
from test_icp_relations import apply_pose, l_shaped_scan


def test_refines_noisy_odometry_to_true_motion():
    scan = l_shaped_scan()
    true_motion = np.array([0.4, -0.2, 0.1])
    # Scan 1 sees the same walls from the moved pose: points in sensor frame.
    c, s = np.cos(true_motion[2]), np.sin(true_motion[2])
    rot = np.array([[c, -s], [s, c]])
    scan1 = (scan - true_motion[:2]) @ rot
    noisy = np.vstack([[0.0, 0.0, 0.0], true_motion + [0.15, -0.1, 0.04]])
    absolute, refined = refine_odometry([scan, scan1], noisy)
    assert refined == 1
    np.testing.assert_allclose(absolute[1], true_motion, atol=1e-5)


def test_falls_back_to_wheel_odometry_without_overlap():
    scan0 = l_shaped_scan()
    scan1 = l_shaped_scan() + np.array([500.0, 500.0])
    poses = np.vstack([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    absolute, refined = refine_odometry([scan0, scan1], poses)
    assert refined == 0
    np.testing.assert_allclose(absolute[1], poses[1], atol=1e-12)

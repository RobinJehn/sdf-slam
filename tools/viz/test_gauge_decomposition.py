"""Unit tests for gauge_decomposition: gap deviations and the global-fit split."""

import numpy as np
import pytest

from gauge_decomposition import gap_deviation, relative_transforms, rigid_residual


def spiral(n: int = 60) -> np.ndarray:
    """Smooth test trajectory: positions on an arc, headings tangent."""
    s = np.linspace(0.0, 3.0, n)
    x = np.cos(s) * (1.0 + s)
    y = np.sin(s) * (1.0 + s)
    theta = s + 0.5
    return np.stack([x, y, theta], axis=1)


def apply_global(poses: np.ndarray, angle: float, t: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    out = poses.copy()
    out[:, :2] = poses[:, :2] @ rot.T + t
    out[:, 2] = poses[:, 2] + angle
    return out


def apply_twist(poses: np.ndarray, rate: float) -> np.ndarray:
    """Rotate pose i about the origin by rate*i: a differential twist."""
    out = poses.copy()
    for i in range(len(poses)):
        angle = rate * i
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        out[i, :2] = rot @ poses[i, :2]
        out[i, 2] = poses[i, 2] + angle
    return out


def test_relative_transforms_shape_and_gap_one_matches_compose():
    poses = spiral()
    rel = relative_transforms(poses, gap=1)
    assert rel.shape == (len(poses) - 1, 3)
    # frame 0 -> frame 1 by hand
    dx, dy = poses[1, :2] - poses[0, :2]
    c, s = np.cos(poses[0, 2]), np.sin(poses[0, 2])
    assert rel[0, 0] == pytest.approx(c * dx + s * dy)
    assert rel[0, 1] == pytest.approx(-s * dx + c * dy)
    assert rel[0, 2] == pytest.approx(poses[1, 2] - poses[0, 2])


def test_gap_deviation_is_invariant_under_global_rigid_motion():
    a = spiral()
    b = apply_global(a, angle=0.3, t=np.array([2.0, -1.0]))
    for gap in (1, 10, 30):
        dev = gap_deviation(a, b, gap)
        assert dev.shape == (len(a) - gap,)
        assert dev.max() == pytest.approx(0.0, abs=1e-9)


def test_gap_deviation_grows_with_gap_under_differential_twist():
    a = spiral()
    b = apply_twist(a, rate=0.002)
    small = gap_deviation(a, b, 1).mean()
    large = gap_deviation(a, b, 20).mean()
    assert large > 5.0 * small


def test_rigid_residual_removes_a_global_motion_entirely():
    a = spiral()
    b = apply_global(a, angle=0.3, t=np.array([2.0, -1.0]))
    raw, internal = rigid_residual(a, b)
    assert raw.mean() > 1.0
    assert internal.max() == pytest.approx(0.0, abs=1e-9)


def test_rigid_residual_cannot_remove_a_differential_twist():
    a = spiral()
    b = apply_twist(a, rate=0.01)
    raw, internal = rigid_residual(a, b)
    assert internal.mean() > 0.2 * raw.mean()

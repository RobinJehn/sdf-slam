"""Unit tests for carmen_to_pcd: CARMEN log parsing, beam geometry, decimation."""

import math

import numpy as np
import pytest

from carmen_to_pcd import anchor_poses, beam_points, decimate, parse_laser_line

FLASER = (
    "FLASER 4 1.0 2.0 3.0 50.0 1.5 2.5 0.75 1.6 2.6 0.85 "
    "70.28 nohost 0.11"
)


def test_parse_laser_line_splits_readings_and_poses():
    scan = parse_laser_line(FLASER)
    assert scan.readings == [1.0, 2.0, 3.0, 50.0]
    assert scan.laser_pose == pytest.approx((1.5, 2.5, 0.75))
    assert scan.odom_pose == pytest.approx((1.6, 2.6, 0.85))


def test_parse_laser_line_rejects_other_messages():
    assert parse_laser_line("ODOM 1 2 3 0 0 0 1.0 nohost 1.0") is None


def test_beam_points_start_at_minus_ninety_degrees():
    points = beam_points([1.09, 2.0, 3.0, 4.0], max_range=49.0, min_range=0.1)
    # first beam looks to the sensor's right: -90 degrees
    assert points[0][0] == pytest.approx(0.0, abs=1e-12)
    assert points[0][1] == pytest.approx(-1.09)


def test_beam_points_span_the_half_circle():
    k = 180
    points = beam_points([5.0] * k, max_range=49.0, min_range=0.1)
    angles = [math.atan2(y, x) for x, y in points]
    assert angles[0] == pytest.approx(-math.pi / 2)
    assert angles[-1] == pytest.approx(-math.pi / 2 + math.pi * (k - 1) / k)


def test_beam_points_drop_out_of_range_readings():
    points = beam_points([1.0, 50.0, 0.0, 2.0], max_range=49.0, min_range=0.1)
    assert len(points) == 2


def test_decimate_keeps_first_and_respects_travel_threshold():
    poses = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.6, 0.0, 0.0), (0.7, 0.0, 0.0)]
    assert decimate(poses, min_travel=0.5, min_turn=10.0) == [0, 2]


def test_decimate_triggers_on_rotation_alone():
    poses = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.0, 0.5)]
    assert decimate(poses, min_travel=99.0, min_turn=0.4) == [0, 2]


def test_decimate_wraps_angles():
    poses = [(0.0, 0.0, math.pi - 0.05), (0.0, 0.0, -math.pi + 0.05)]
    # the two headings are 0.1 rad apart, not 2*pi - 0.1
    assert decimate(poses, min_travel=99.0, min_turn=0.5) == [0]


def test_anchor_poses_puts_the_first_scan_at_the_origin():
    poses = [(3.0, 4.0, math.pi / 2), (3.0, 5.0, math.pi / 2)]
    anchored = anchor_poses(poses)
    assert anchored[0] == pytest.approx((0.0, 0.0, 0.0))
    # the second pose lay one metre along the first pose's +y (its left)
    assert anchored[1] == pytest.approx((1.0, 0.0, 0.0))


def test_anchor_poses_preserves_relative_geometry():
    poses = [(3.0, 4.0, 0.3), (5.0, 4.5, 0.8), (6.0, 7.0, -0.2)]
    anchored = np.array(anchor_poses(poses))
    raw = np.array(poses)
    for i in range(len(poses) - 1):
        assert np.linalg.norm(anchored[i + 1, :2] - anchored[i, :2]) == pytest.approx(
            np.linalg.norm(raw[i + 1, :2] - raw[i, :2])
        )


def test_relation_times_reads_both_endpoint_columns(tmp_path):
    from carmen_to_pcd import relation_times

    path = tmp_path / "x.relations"
    path.write_text("1.5 9.5 0.1 0.2 0 0 0 0.3\n2.5 1.5 0.4 0.5 0 0 0 0.6\n")
    assert relation_times(path) == {1.5, 2.5, 9.5}


def test_index_relations_maps_timestamps_to_scan_indices(tmp_path):
    from carmen_to_pcd import index_relations

    path = tmp_path / "x.relations"
    path.write_text("1.5 9.5 0.1 0.2 0.0 0.0 0.0 0.3\n")
    rows = index_relations(path, {1.5: 0, 9.5: 7})
    assert rows == [(0, 7, 0.1, 0.2, 0.3)]


def test_index_relations_skips_pairs_whose_scan_was_dropped(tmp_path):
    from carmen_to_pcd import index_relations

    path = tmp_path / "x.relations"
    path.write_text("1.5 9.5 0.1 0.2 0.0 0.0 0.0 0.3\n2.5 9.5 0.4 0.5 0.0 0.0 0.0 0.6\n")
    rows = index_relations(path, {1.5: 0, 9.5: 7})
    assert len(rows) == 1

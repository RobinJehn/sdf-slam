#!/usr/bin/env python3
"""Convert a CARMEN logfile into the dataset layout this project reads.

CARMEN logs (the Radish / Freiburg SLAM-evaluation `.clf` files: ACES,
MIT Killian Court, fr079) carry one text line per message. This tool reads
the laser messages (`FLASER`, and `ROBOTLASER1` whose readings sit after a
config block) and writes the layout of `data/intel`: one ASCII `.pcd` per
scan holding the beam endpoints in the sensor frame, plus a
`scanner_info.txt` with one `x y theta` odometry pose per scan.

Logs are recorded far denser than the solver needs, so scans are decimated
by travelled distance and turn angle. Poses are re-anchored so scan 0 sits
at the origin, matching the frame-0 anchor of the solver (DEC-0007).

Usage: uv run carmen_to_pcd.py <log.clf> --out data/aces [--min-travel 0.5]
see DEC-0009 parameter-selection-by-ensembles
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

LASER_MESSAGES = ("FLASER", "ROBOTLASER1")


@dataclass
class LaserScan:
    """One laser message: ranges in metres plus the two poses it carries."""

    readings: list[float]
    laser_pose: tuple[float, float, float]
    odom_pose: tuple[float, float, float]
    timestamp: float


def parse_laser_line(line: str) -> LaserScan | None:
    """Parse one CARMEN laser message, or return None for other messages."""
    parts = line.split()
    if not parts or parts[0] not in LASER_MESSAGES:
        return None
    if parts[0] == "FLASER":
        count = int(parts[1])
        first = 2
    else:
        # ROBOTLASER1 prefixes a config block: type, start angle, fov,
        # resolution, max range, accuracy, remission mode, then the count.
        count = int(parts[8])
        first = 9
    readings = [float(v) for v in parts[first:first + count]]
    rest = parts[first + count:]
    laser = (float(rest[0]), float(rest[1]), float(rest[2]))
    odom = (float(rest[3]), float(rest[4]), float(rest[5]))
    return LaserScan(readings, laser, odom, float(rest[6]))


def beam_points(readings: list[float], max_range: float,
                min_range: float) -> list[tuple[float, float]]:
    """Beam endpoints in the sensor frame.

    CARMEN laser messages span a 180 degree field of view. Beam i points at
    -90 degrees + i * 180 / count degrees. Readings at or beyond `max_range`
    are the sensor's no-return sentinel and carry no surface.
    """
    count = len(readings)
    step = math.pi / count
    points = []
    for i, r in enumerate(readings):
        if r < min_range or r >= max_range:
            continue
        angle = -math.pi / 2 + i * step
        points.append((r * math.cos(angle), r * math.sin(angle)))
    return points


def decimate(poses: list[tuple[float, float, float]], min_travel: float,
             min_turn: float) -> list[int]:
    """Indices of the scans to keep, thinned by travelled distance and turn."""
    if not poses:
        return []
    kept = [0]
    last = poses[0]
    for i, pose in enumerate(poses[1:], start=1):
        moved = math.hypot(pose[0] - last[0], pose[1] - last[1])
        turned = abs(math.atan2(math.sin(pose[2] - last[2]),
                                math.cos(pose[2] - last[2])))
        if moved >= min_travel or turned >= min_turn:
            kept.append(i)
            last = pose
    return kept


def anchor_poses(
    poses: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Express every pose in the first pose's frame, so scan 0 is the origin."""
    x0, y0, t0 = poses[0]
    c, s = math.cos(t0), math.sin(t0)
    anchored = []
    for x, y, t in poses:
        dx, dy = x - x0, y - y0
        angle = math.atan2(math.sin(t - t0), math.cos(t - t0))
        anchored.append((c * dx + s * dy, -s * dx + c * dy, angle))
    return anchored


def relation_times(path: Path) -> set[float]:
    """Every scan timestamp a benchmark relations file refers to."""
    times = set()
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            times.update((float(parts[0]), float(parts[1])))
    return times


def index_relations(path: Path,
                    time_to_index: dict[float, int]) -> list[tuple]:
    """Benchmark relations as (i, j, dx, dy, dtheta) over kept scan indices.

    The Freiburg SLAM-evaluation relations files list a relative pose per
    scan pair, keyed by the log timestamps. Pairs whose scans the
    decimation dropped cannot be scored and are skipped.
    """
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        start, end = float(parts[0]), float(parts[1])
        if start not in time_to_index or end not in time_to_index:
            continue
        rows.append((time_to_index[start], time_to_index[end],
                     float(parts[2]), float(parts[3]), float(parts[7])))
    return rows


def write_pcd(path: Path, points: list[tuple[float, float]]) -> None:
    """Write the ASCII PCD dialect the scan loader reads (x y, no header z)."""
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y\nSIZE 4 4\nTYPE F F\nCOUNT 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\nDATA ascii\n"
    )
    body = "".join(f"{x!r} {y!r}\n" for x, y in points)
    path.write_text(header + body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-travel", type=float, default=0.5,
                        help="metres of travel between kept scans")
    parser.add_argument("--min-turn", type=float, default=0.2,
                        help="radians of rotation between kept scans")
    parser.add_argument("--max-range", type=float, default=49.0,
                        help="readings at or beyond this are no-return")
    parser.add_argument("--min-range", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many kept scans (0 = all)")
    parser.add_argument("--relations", type=Path, default=None,
                        help="benchmark relations file; its scans are always "
                             "kept and it is rewritten with scan indices")
    args = parser.parse_args()

    scans = []
    with args.log.open() as handle:
        for line in handle:
            scan = parse_laser_line(line)
            if scan is not None:
                scans.append(scan)
    if not scans:
        raise SystemExit(f"no laser messages in {args.log}")

    keep = decimate([s.laser_pose for s in scans], args.min_travel, args.min_turn)
    if args.relations:
        wanted = relation_times(args.relations)
        keep = sorted(set(keep) | {i for i, s in enumerate(scans)
                                   if s.timestamp in wanted})
    if args.limit:
        keep = keep[:args.limit]
    poses = anchor_poses([scans[i].laser_pose for i in keep])

    args.out.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(keep) - 1)))
    for slot, index in enumerate(keep):
        points = beam_points(scans[index].readings, args.max_range, args.min_range)
        write_pcd(args.out / f"scan{slot:0{width}d}.pcd", points)
    (args.out / "scanner_info.txt").write_text(
        "".join(f"{x!r} {y!r} {t!r}\n" for x, y, t in poses)
    )
    print(f"{len(scans)} laser messages -> {len(keep)} scans in {args.out}")

    if args.relations:
        time_to_index = {scans[index].timestamp: slot
                         for slot, index in enumerate(keep)}
        rows = index_relations(args.relations, time_to_index)
        target = args.out.parent / f"{args.out.name}_relations.csv"
        target.write_text(
            "i,j,dx,dy,dtheta,residual,inliers\n"
            + "".join(f"{i},{j},{dx!r},{dy!r},{dt!r},0.0,0\n"
                      for i, j, dx, dy, dt in rows)
        )
        print(f"{len(rows)} benchmark relations -> {target}")


if __name__ == "__main__":
    main()

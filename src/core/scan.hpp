#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <vector>

#include "core/pose.hpp"

namespace sdf_slam {

/// A single LiDAR scan. Points are in the sensor frame.
struct Scan {
  std::vector<Eigen::Vector2d> points;
};

/// A sequence of scans with per-scan absolute poses from scanner_info.txt.
/// For simulated data the poses are ground truth; for real data they are
/// integrated odometry.
struct Dataset {
  std::vector<Scan> scans;
  std::vector<Pose2> poses;
};

/// Parses an ASCII PCD file with at least x and y fields.
/// Throws std::runtime_error on malformed or binary files.
Scan LoadPcd(const std::filesystem::path& path);

/// Loads scan*.pcd files (sorted by filename) and scanner_info.txt from a
/// directory. Throws if the scan count and the pose count differ.
Dataset LoadDataset(const std::filesystem::path& dir);

/// Loads poses from a scanner_info.txt-style file (rows of "x y theta").
std::vector<Pose2> LoadPoses(const std::filesystem::path& path);

}  // namespace sdf_slam

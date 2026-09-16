#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <filesystem>

#include "problem/problem.hpp"
#include "solvers/solver.hpp"

namespace sdf_slam {

enum class RunMode : std::uint8_t { kBatch, kIncremental };

struct MapConfig {
  int nx{100};
  int ny{100};
  /// Fixed domain bounds; ignored when auto_domain is set.
  Eigen::Vector2d min_corner{-25.0, -25.0};
  Eigen::Vector2d max_corner{25.0, 25.0};
  /// Derive the domain from the data bounds plus margin.
  bool auto_domain{false};
  double auto_domain_margin{5.0};
  double initial_value{0.0};
  /// Seed every node with the signed distance to the nearest scan point
  /// instead of `initial_value`. Batch runs seed from all scans at their
  /// starting poses; incremental runs seed from frame 0 only, so the map
  /// never carries knowledge of scans the run has not reached yet.
  bool init_from_scans{false};
};

struct RunConfig {
  std::filesystem::path dataset_dir;
  /// Ground-truth poses for evaluation; empty when unavailable.
  std::filesystem::path ground_truth_poses;
  /// Batch mode: start the optimization from these poses instead of the
  /// dataset odometry (e.g. a previous run's poses_estimated.csv for a global
  /// polish pass). Odometry residual measurements still come from the dataset.
  std::filesystem::path initial_poses;
  /// Relative poses between arbitrary frame pairs, added as relation
  /// residuals with weights.relation (CSV: header line, then
  /// i,j,dx,dy,dtheta[,...] — the icp_relations.py output format).
  std::filesystem::path relations_file;
  std::filesystem::path output_dir{"out"};

  RunMode mode{RunMode::kBatch};
  /// Incremental mode: frames appended per step (algorithm 2, increment k).
  int increment_size{1};
  /// Incremental mode: solver iterations per step.
  int iterations_per_increment{10};
  /// Incremental mode: write map + poses to output_dir/snapshots every N
  /// increments so tools can animate the run. 0 disables snapshots.
  int snapshot_every{0};

  MapConfig map;
  ProblemOptions problem;
  SolverOptions solver;
};

/// Parses a YAML run configuration. Unknown keys raise, so typos in
/// experiment configs fail loudly.
RunConfig LoadConfig(const std::filesystem::path& path);

}  // namespace sdf_slam

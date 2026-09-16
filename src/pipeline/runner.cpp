#include "pipeline/runner.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "core/map_init.hpp"
#include "core/scan.hpp"
#include "problem/problem.hpp"
#include "solvers/solver.hpp"

namespace sdf_slam {

std::vector<RelationMeasurement> LoadRelations(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open relations file: " + path.string());
  }
  std::vector<RelationMeasurement> relations;
  std::string line;
  while (std::getline(file, line)) {
    std::ranges::replace(line, ',', ' ');
    std::istringstream stream(line);
    RelationMeasurement rel;
    double dx = 0.0;
    double dy = 0.0;
    double dtheta = 0.0;
    if (stream >> rel.frame_i >> rel.frame_j >> dx >> dy >> dtheta) {
      rel.measurement = {dx, dy, dtheta};
      double trailing = 0.0;
      stream >> trailing >> trailing;  // residual and inliers, ignored
      double weight = 1.0;
      if (stream >> weight) {
        if (weight <= 0.0) {
          throw std::runtime_error("relation weight must be positive in " + path.string());
        }
        rel.sqrt_weight = std::sqrt(weight);
      }
      relations.push_back(rel);
    }
  }
  return relations;
}

namespace {

std::vector<Pose2> RelativeOdometry(const std::vector<Pose2>& poses) {
  std::vector<Pose2> odometry;
  odometry.reserve(poses.size() - 1);
  for (size_t i = 0; i + 1 < poses.size(); ++i) {
    odometry.push_back(poses[i].RelativeTo(poses[i + 1]));
  }
  return odometry;
}

GridMap BuildMap(const MapConfig& config, const Dataset& dataset) {
  if (!config.auto_domain) {
    return {config.nx, config.ny, config.min_corner, config.max_corner, config.initial_value};
  }
  Eigen::Vector2d min_corner(std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max());
  Eigen::Vector2d max_corner = -min_corner;
  for (size_t i = 0; i < dataset.scans.size(); ++i) {
    for (const auto& p : dataset.scans[i].points) {
      const Eigen::Vector2d global = dataset.poses[i].Apply(p);
      min_corner = min_corner.cwiseMin(global);
      max_corner = max_corner.cwiseMax(global);
    }
    min_corner = min_corner.cwiseMin(dataset.poses[i].Translation());
    max_corner = max_corner.cwiseMax(dataset.poses[i].Translation());
  }
  min_corner.array() -= config.auto_domain_margin;
  max_corner.array() += config.auto_domain_margin;
  return {config.nx, config.ny, min_corner, max_corner, config.initial_value};
}

void WritePoses(const std::filesystem::path& path, const std::vector<Pose2>& poses) {
  std::ofstream file(path);
  file << std::setprecision(17) << "x,y,theta\n";
  for (const auto& pose : poses) {
    file << pose.x << "," << pose.y << "," << pose.theta << "\n";
  }
}

void WriteMap(const std::filesystem::path& path, const GridMap& map) {
  std::ofstream file(path);
  file << std::setprecision(17) << map.nx() << " " << map.ny() << " " << map.min_corner().x() << " "
       << map.min_corner().y() << " " << map.dx() << " " << map.dy() << "\n";
  for (int h = 0; h < map.ny(); ++h) {
    for (int w = 0; w < map.nx(); ++w) {
      file << map.Value(w, h) << (w + 1 < map.nx() ? " " : "\n");
    }
  }
}

/// Writes every scan point in the global frame, one row per point, so plots
/// can overlay the scans on the map without access to the dataset.
void WriteGlobalScanPoints(const std::filesystem::path& path, const std::vector<Scan>& scans,
                           const std::vector<Pose2>& poses) {
  std::ofstream file(path);
  file << std::setprecision(9) << "x,y\n";
  for (size_t i = 0; i < scans.size(); ++i) {
    for (const auto& p : scans[i].points) {
      const Eigen::Vector2d global = poses[i].Apply(p);
      file << global.x() << "," << global.y() << "\n";
    }
  }
}

/// Scan points outside the map domain evaluate the SDF as 0 with no gradient,
/// so a large out-of-domain fraction weakens the optimization. Warn loudly.
void WarnIfPointsOutsideMap(const GridMap& map, const std::vector<Scan>& scans,
                            const std::vector<Pose2>& poses) {
  size_t outside = 0;
  size_t total = 0;
  GridMap::CellRef cell;
  for (size_t i = 0; i < scans.size(); ++i) {
    for (const auto& p : scans[i].points) {
      if (!map.Locate(poses[i].Apply(p), cell)) {
        ++outside;
      }
      ++total;
    }
  }
  if (outside > 0) {
    std::cerr << "WARNING: " << outside << " of " << total << " scan points ("
              << 100.0 * static_cast<double>(outside) / static_cast<double>(total)
              << "%) fall outside the map domain and are unanchored (SDF 0, no gradient). "
                 "Widen the map domain or disable auto_domain with explicit corners.\n";
  }
}

struct TrajectoryMetrics {
  double mean_translation_error{0.0};
  double mean_rotation_error{0.0};
  double mean_relative_translation_error{0.0};
  double mean_relative_rotation_error{0.0};
};

TrajectoryMetrics Evaluate(const std::vector<Pose2>& estimated,
                           const std::vector<Pose2>& ground_truth) {
  if (estimated.size() != ground_truth.size() || estimated.size() < 2) {
    throw std::runtime_error("pose count mismatch between estimate and ground truth");
  }
  TrajectoryMetrics metrics;
  for (size_t i = 0; i < estimated.size(); ++i) {
    metrics.mean_translation_error +=
        (estimated[i].Translation() - ground_truth[i].Translation()).norm();
    metrics.mean_rotation_error += std::abs(WrapAngle(estimated[i].theta - ground_truth[i].theta));
  }
  metrics.mean_translation_error /= static_cast<double>(estimated.size());
  metrics.mean_rotation_error /= static_cast<double>(estimated.size());

  const size_t pairs = estimated.size() - 1;
  for (size_t i = 0; i < pairs; ++i) {
    const Pose2 rel_est = estimated[i].RelativeTo(estimated[i + 1]);
    const Pose2 rel_gt = ground_truth[i].RelativeTo(ground_truth[i + 1]);
    metrics.mean_relative_translation_error +=
        (rel_est.Translation() - rel_gt.Translation()).norm();
    metrics.mean_relative_rotation_error += std::abs(WrapAngle(rel_est.theta - rel_gt.theta));
  }
  metrics.mean_relative_translation_error /= static_cast<double>(pairs);
  metrics.mean_relative_rotation_error /= static_cast<double>(pairs);
  return metrics;
}

void WriteJson(const std::filesystem::path& path, const SolveResult& result,
               const TrajectoryMetrics* metrics) {
  std::ofstream file(path);
  file << std::setprecision(17) << "{\n";
  file << "  \"converged\": " << (result.converged ? "true" : "false") << ",\n";
  file << "  \"iterations\": " << result.iterations << ",\n";
  file << "  \"initial_cost\": " << result.initial_cost << ",\n";
  file << "  \"final_cost\": " << result.final_cost << ",\n";
  file << "  \"duration_seconds\": " << result.duration_seconds;
  if (metrics != nullptr) {
    file << ",\n";
    file << "  \"mean_translation_error\": " << metrics->mean_translation_error << ",\n";
    file << "  \"mean_rotation_error\": " << metrics->mean_rotation_error << ",\n";
    file << "  \"mean_relative_translation_error\": " << metrics->mean_relative_translation_error
         << ",\n";
    file << "  \"mean_relative_rotation_error\": " << metrics->mean_relative_rotation_error << "\n";
  } else {
    file << "\n";
  }
  file << "}\n";
}

}  // namespace

SolveResult Run(const RunConfig& config) {
  const Dataset dataset = LoadDataset(config.dataset_dir);
  if (dataset.scans.size() < 2) {
    throw std::runtime_error("dataset needs at least two scans");
  }
  const std::vector<Pose2> odometry = RelativeOdometry(dataset.poses);
  std::vector<RelationMeasurement> relations;
  if (!config.relations_file.empty()) {
    relations = LoadRelations(config.relations_file);
  }
  GridMap map = BuildMap(config.map, dataset);

  SolveResult total;
  std::vector<Pose2> estimated;

  if (config.mode == RunMode::kBatch) {
    std::vector<Pose2> initial = dataset.poses;
    if (!config.initial_poses.empty()) {
      initial = LoadPoses(config.initial_poses);
      if (initial.size() != dataset.scans.size()) {
        throw std::runtime_error("initial_poses count != scan count");
      }
    }
    if (config.map.init_from_scans) {
      InitializeFromScans(map, dataset.scans, initial);
    }
    Problem problem(std::move(map), initial, dataset.scans, odometry, config.problem, relations);
    total = Solve(problem, config.solver);
    estimated = problem.poses();
    map = problem.map();
  } else {
    // Incremental optimization (dissertation algorithm 2): append frames by
    // composing odometry onto the last optimized pose, then refine everything
    // added so far.
    estimated = {dataset.poses[0]};
    if (config.map.init_from_scans) {
      InitializeFromScans(map, {dataset.scans[0]}, {dataset.poses[0]});
    }
    SolverOptions step_options = config.solver;
    step_options.max_iterations = config.iterations_per_increment;
    total.initial_cost = -1.0;

    const std::filesystem::path snapshot_dir = config.output_dir / "snapshots";
    if (config.snapshot_every > 0) {
      std::filesystem::create_directories(snapshot_dir);
    }
    size_t increment = 0;

    size_t next_frame = 1;
    while (next_frame < dataset.scans.size()) {
      const size_t last =
          std::min(dataset.scans.size(), next_frame + static_cast<size_t>(config.increment_size));
      for (size_t i = next_frame; i < last; ++i) {
        estimated.push_back(estimated.back().Compose(odometry[i - 1]));
      }
      next_frame = last;

      const std::vector<Scan> scans_so_far(
          dataset.scans.begin(), dataset.scans.begin() + static_cast<std::ptrdiff_t>(next_frame));
      const std::vector<Pose2> odom_so_far(
          odometry.begin(), odometry.begin() + static_cast<std::ptrdiff_t>(next_frame - 1));
      Problem problem(std::move(map), estimated, scans_so_far, odom_so_far, config.problem,
                      relations);
      const SolveResult step = Solve(problem, step_options);
      estimated = problem.poses();
      map = problem.map();

      if (total.initial_cost < 0.0) {
        total.initial_cost = step.initial_cost;
      }
      total.iterations += step.iterations;
      total.final_cost = step.final_cost;
      total.duration_seconds += step.duration_seconds;
      total.converged = step.converged;
      total.cost_history.insert(total.cost_history.end(), step.cost_history.begin(),
                                step.cost_history.end());

      ++increment;
      if (config.snapshot_every > 0 &&
          (increment % static_cast<size_t>(config.snapshot_every) == 0 ||
           next_frame == dataset.scans.size())) {
        // File names carry the number of frames in the snapshot so the
        // animation can pair each map with its poses.
        std::ostringstream tag;
        tag << std::setw(6) << std::setfill('0') << next_frame;
        WriteMap(snapshot_dir / ("map_" + tag.str() + ".txt"), map);
        WritePoses(snapshot_dir / ("poses_" + tag.str() + ".csv"), estimated);
      }
    }
  }

  std::filesystem::create_directories(config.output_dir);
  WritePoses(config.output_dir / "poses_odometry.csv", dataset.poses);
  WritePoses(config.output_dir / "poses_estimated.csv", estimated);
  WriteMap(config.output_dir / "map.txt", map);
  WriteGlobalScanPoints(config.output_dir / "scan_points_global.csv", dataset.scans, estimated);
  WarnIfPointsOutsideMap(map, dataset.scans, estimated);

  if (!config.ground_truth_poses.empty()) {
    const std::vector<Pose2> ground_truth = LoadPoses(config.ground_truth_poses);
    WritePoses(config.output_dir / "poses_ground_truth.csv", ground_truth);
    const TrajectoryMetrics metrics = Evaluate(estimated, ground_truth);
    WriteJson(config.output_dir / "metrics.json", total, &metrics);
  } else {
    WriteJson(config.output_dir / "metrics.json", total, nullptr);
  }
  return total;
}

}  // namespace sdf_slam

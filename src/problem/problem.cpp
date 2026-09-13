#include "problem/problem.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "problem/residual_math.hpp"

namespace sdf_slam {

Problem::Problem(GridMap map, std::vector<Pose2> poses, const std::vector<Scan>& scans,
                 const std::vector<Pose2>& odometry, const ProblemOptions& options)
    : map_(std::move(map)), poses_(std::move(poses)) {
  if (scans.size() != poses_.size()) {
    throw std::invalid_argument("scan count != pose count");
  }
  if (!odometry.empty() && odometry.size() != scans.size() - 1) {
    throw std::invalid_argument("odometry count must be frames - 1");
  }

  // Point residuals: scans plus hallucinated points.
  const double sqrt_ws = std::sqrt(options.weights.scan);
  const double sqrt_wh = std::sqrt(options.weights.hallucination);
  for (size_t i = 0; i < scans.size(); ++i) {
    const int frame = static_cast<int>(i);
    for (const auto& p : scans[i].points) {
      point_specs_.push_back({frame, p, 0.0, sqrt_ws});
    }
    for (const auto& hall : GenerateHallucinatedPoints(scans[i], options.hallucination)) {
      point_specs_.push_back({frame, hall.point_sensor, hall.expected_sdf, sqrt_wh});
    }
  }

  // Active node set: nodes of cells touched by point residuals at the initial
  // poses, dilated by active_margin cells so points can move during the solve
  // and Eikonal residuals cover the surroundings.
  // see DEC-0002 active-region-map
  node_to_active_.assign(static_cast<size_t>(map_.num_nodes()), -1);
  std::vector<char> active(static_cast<size_t>(map_.num_nodes()), 0);
  if (options.active_region) {
    const int margin = options.active_margin;
    for (const auto& spec : point_specs_) {
      const Eigen::Vector2d global =
          poses_[static_cast<size_t>(spec.frame)].Apply(spec.point_sensor);
      GridMap::CellRef cell;
      if (!map_.Locate(global, cell)) {
        continue;
      }
      for (int h = std::max(0, cell.h - margin); h <= std::min(map_.ny() - 1, cell.h + 1 + margin);
           ++h) {
        for (int w = std::max(0, cell.w - margin);
             w <= std::min(map_.nx() - 1, cell.w + 1 + margin); ++w) {
          active[static_cast<size_t>(map_.NodeId(w, h))] = 1;
        }
      }
    }
  } else {
    std::ranges::fill(active, 1);
  }
  for (int node = 0; node < map_.num_nodes(); ++node) {
    if (active[static_cast<size_t>(node)] != 0) {
      node_to_active_[static_cast<size_t>(node)] = static_cast<int>(active_to_node_.size());
      active_to_node_.push_back(node);
    }
  }

  // Eikonal residuals at every active node whose forward-difference neighbors
  // are also active (dissertation eq. 3.20; last row/column carries none).
  const ScanNormals scan_normals = ComputeScanNormals(scans, poses_, options.normals.k_neighbors);
  const std::vector<NodeNormal> node_normals =
      AssignNodeNormals(map_, scan_normals, options.normals);
  const double sqrt_we = std::sqrt(options.weights.eikonal);
  for (int h = 0; h + 1 < map_.ny(); ++h) {
    for (int w = 0; w + 1 < map_.nx(); ++w) {
      const int node = map_.NodeId(w, h);
      if (node_to_active_[static_cast<size_t>(node)] < 0 ||
          node_to_active_[static_cast<size_t>(map_.NodeId(w + 1, h))] < 0 ||
          node_to_active_[static_cast<size_t>(map_.NodeId(w, h + 1))] < 0) {
        continue;
      }
      const NodeNormal& nn = node_normals[static_cast<size_t>(node)];
      const double scale = nn.eikonal_scale;
      if (scale <= 0.0) {
        continue;
      }
      eikonal_specs_.push_back({w, h, nn.normal, sqrt_we * std::sqrt(scale)});
    }
  }

  // Odometry residuals between consecutive frames.
  const double sqrt_wo = std::sqrt(options.weights.odometry);
  for (size_t i = 0; i + 1 < scans.size() && i < odometry.size(); ++i) {
    odom_specs_.push_back({static_cast<int>(i), odometry[i], sqrt_wo});
  }
}

void Problem::ApplyStep(const Eigen::VectorXd& step) {
  if (step.size() != num_parameters()) {
    throw std::invalid_argument("step size != parameter count");
  }
  for (int a = 0; a < num_active_nodes(); ++a) {
    map_.values()[active_to_node_[static_cast<size_t>(a)]] += step[a];
  }
  for (size_t i = 1; i < poses_.size(); ++i) {
    const int col = PoseColumn(static_cast<int>(i));
    poses_[i].x += step[col];
    poses_[i].y += step[col + 1];
    poses_[i].theta = WrapAngle(poses_[i].theta + step[col + 2]);
  }
}

double Problem::Cost() const {
  double cost = 0.0;
  for (const auto& spec : point_specs_) {
    const PointResidualJacobian eval = EvalPointResidual(
        map_, poses_[static_cast<size_t>(spec.frame)], spec.point_sensor, spec.expected_sdf);
    const double r = spec.sqrt_weight * eval.residual;
    cost += r * r;
  }
  for (const auto& spec : eikonal_specs_) {
    const EikonalResidualJacobian eval = EvalEikonalResidual(map_, spec.w, spec.h, spec.normal);
    const double r = spec.sqrt_weight * eval.residual;
    cost += r * r;
  }
  for (const auto& spec : odom_specs_) {
    const OdomResidualJacobian eval =
        EvalOdomResidual(poses_[static_cast<size_t>(spec.frame_i)],
                         poses_[static_cast<size_t>(spec.frame_i) + 1], spec.measurement);
    cost += (spec.sqrt_weight * eval.residual).squaredNorm();
  }
  return cost;
}

}  // namespace sdf_slam

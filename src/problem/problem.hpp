#pragma once

#include <Eigen/Core>
#include <optional>
#include <vector>

#include "core/grid_map.hpp"
#include "core/hallucination.hpp"
#include "core/normals.hpp"
#include "core/pose.hpp"
#include "core/scan.hpp"

namespace sdf_slam {

/// Residual weights of the objective (dissertation eq. 3.11).
struct Weights {
  double scan{1.0};
  double hallucination{1.0};
  double eikonal{1.0};
  double odometry{1.0};
};

/// A scan or hallucinated point residual: r = D(T_frame p) - expected_sdf.
struct PointSpec {
  int frame{0};
  Eigen::Vector2d point_sensor{0.0, 0.0};
  double expected_sdf{0.0};
  double sqrt_weight{1.0};
};

/// An Eikonal residual at grid node (w, h).
struct EikonalSpec {
  int w{0};
  int h{0};
  Eigen::Vector2d normal{0.0, 0.0};
  double sqrt_weight{1.0};
};

/// A relative odometry residual between frames i and i+1.
struct OdomSpec {
  int frame_i{0};
  Pose2 measurement;
  double sqrt_weight{1.0};
};

struct ProblemOptions {
  Weights weights;
  HallucinationOptions hallucination;
  NormalOptions normals;
  /// When true, only grid nodes referenced by point residuals (dilated by
  /// active_margin cells) enter the state vector; the rest stay constant.
  /// see DEC-0002 active-region-map
  bool active_region{false};
  int active_margin{2};
};

/// The joint SLAM optimization problem over map values and poses. Frame 0 is
/// fixed; the state comprises the (active) grid values and poses 1..N-1
/// (dissertation eq. 3.9).
class Problem {
 public:
  /// Builds the problem for the given frames. `odometry` holds the relative
  /// measurement from frame i to i+1 (size = frames - 1). Normals and
  /// hallucinated points are generated from the current poses.
  Problem(GridMap map, std::vector<Pose2> poses, const std::vector<Scan>& scans,
          const std::vector<Pose2>& odometry, const ProblemOptions& options);

  [[nodiscard]] const GridMap& map() const { return map_; }
  GridMap& map() { return map_; }
  [[nodiscard]] const std::vector<Pose2>& poses() const { return poses_; }
  std::vector<Pose2>& poses() { return poses_; }

  [[nodiscard]] const std::vector<PointSpec>& point_specs() const { return point_specs_; }
  [[nodiscard]] const std::vector<EikonalSpec>& eikonal_specs() const { return eikonal_specs_; }
  [[nodiscard]] const std::vector<OdomSpec>& odom_specs() const { return odom_specs_; }

  /// State layout: [active node values; pose_1; ...; pose_{N-1}].
  [[nodiscard]] int num_active_nodes() const { return static_cast<int>(active_to_node_.size()); }
  [[nodiscard]] int num_parameters() const {
    return num_active_nodes() + 3 * (static_cast<int>(poses_.size()) - 1);
  }
  [[nodiscard]] int num_residuals() const {
    return static_cast<int>(point_specs_.size() + eikonal_specs_.size() + 3 * odom_specs_.size());
  }

  /// Column of a node in the state, or -1 when the node is constant.
  [[nodiscard]] int NodeColumn(int node_id) const {
    return node_to_active_[static_cast<size_t>(node_id)];
  }
  /// First column of pose i (i >= 1).
  [[nodiscard]] int PoseColumn(int frame) const { return num_active_nodes() + 3 * (frame - 1); }
  [[nodiscard]] int ActiveNode(int active_index) const {
    return active_to_node_[static_cast<size_t>(active_index)];
  }

  /// Applies a state increment: map values of active nodes and poses 1..N-1.
  void ApplyStep(const Eigen::VectorXd& step);

  /// Total weighted squared error at the current state.
  [[nodiscard]] double Cost() const;

 private:
  GridMap map_;
  std::vector<Pose2> poses_;
  std::vector<PointSpec> point_specs_;
  std::vector<EikonalSpec> eikonal_specs_;
  std::vector<OdomSpec> odom_specs_;
  std::vector<int> node_to_active_;
  std::vector<int> active_to_node_;
};

}  // namespace sdf_slam

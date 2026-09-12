#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>

#include "core/grid_map.hpp"
#include "core/pose.hpp"

namespace sdf_slam {

/// Shared analytic residual/Jacobian math for both solver backends
/// (dissertation sections 3.3.1-3.3.4).
/// see DEC-0001 dual-solver-backends

/// Residual r = D(T p) - delta for a scan point (delta = 0) or a hallucinated
/// point (delta = expected SDF). Jacobian entries cover the four cell nodes
/// and the pose (tx, ty, theta).
struct PointResidualJacobian {
  bool valid{false};
  double residual{0.0};
  std::array<int, 4> node_ids{};
  std::array<double, 4> d_nodes{};
  Eigen::Vector3d d_pose{0.0, 0.0, 0.0};
};

inline PointResidualJacobian EvalPointResidual(const GridMap& map, const Pose2& pose,
                                               const Eigen::Vector2d& point_sensor, double delta) {
  PointResidualJacobian out;
  const Eigen::Vector2d point_global = pose.Apply(point_sensor);
  GridMap::CellRef cell;
  if (!map.Locate(point_global, cell)) {
    return out;
  }
  out.valid = true;
  out.residual = map.Interpolate(cell) - delta;
  out.node_ids = cell.node_ids;
  out.d_nodes = cell.weights;

  const Eigen::Vector2d grad = map.Gradient(cell);
  const double c = std::cos(pose.theta);
  const double s = std::sin(pose.theta);
  // d(point_global)/dtheta (eq. 3.35).
  const double dx_dtheta = -s * point_sensor.x() - c * point_sensor.y();
  const double dy_dtheta = c * point_sensor.x() - s * point_sensor.y();
  out.d_pose = {grad.x(), grad.y(), grad.x() * dx_dtheta + grad.y() * dy_dtheta};
  return out;
}

/// Evaluates the same point residual but with a frozen cell: alpha/beta are
/// recomputed from the current pose while the four nodes stay those chosen at
/// problem-build time. Used by the Ceres backend, whose parameter blocks are
/// fixed per solve; a point that leaves its build-time cell extrapolates that
/// cell's bilinear patch. see DEC-0004 ceres-frozen-cells
inline PointResidualJacobian EvalPointResidualFrozenCell(const GridMap& map, int cell_w, int cell_h,
                                                         const Pose2& pose,
                                                         const Eigen::Vector2d& point_sensor,
                                                         double delta,
                                                         const std::array<double, 4>& node_values) {
  PointResidualJacobian out;
  const Eigen::Vector2d point_global = pose.Apply(point_sensor);
  const Eigen::Vector2d origin = map.NodePosition(cell_w, cell_h);
  const double alpha = (point_global.x() - origin.x()) / map.dx();
  const double beta = (point_global.y() - origin.y()) / map.dy();

  const std::array<double, 4> weights = {(1.0 - alpha) * (1.0 - beta), alpha * (1.0 - beta),
                                         (1.0 - alpha) * beta, alpha * beta};
  out.valid = true;
  out.residual = -delta;
  for (size_t i = 0; i < 4; ++i) {
    out.residual += weights[i] * node_values[i];
  }
  out.d_nodes = weights;
  out.node_ids = {map.NodeId(cell_w, cell_h), map.NodeId(cell_w + 1, cell_h),
                  map.NodeId(cell_w, cell_h + 1), map.NodeId(cell_w + 1, cell_h + 1)};

  const double gx = ((1.0 - beta) * (node_values[1] - node_values[0]) +
                     beta * (node_values[3] - node_values[2])) /
                    map.dx();
  const double gy = ((1.0 - alpha) * (node_values[2] - node_values[0]) +
                     alpha * (node_values[3] - node_values[1])) /
                    map.dy();
  const double c = std::cos(pose.theta);
  const double s = std::sin(pose.theta);
  const double dx_dtheta = -s * point_sensor.x() - c * point_sensor.y();
  const double dy_dtheta = c * point_sensor.x() - s * point_sensor.y();
  out.d_pose = {gx, gy, gx * dx_dtheta + gy * dy_dtheta};
  return out;
}

/// Eikonal residual at node (w, h) with forward differences (eq. 3.18-3.19):
/// r = 1 - n . grad(D). Jacobian entries for nodes (w,h), (w+1,h), (w,h+1)
/// (eq. 3.40-3.42).
struct EikonalResidualJacobian {
  double residual{0.0};
  std::array<int, 3> node_ids{};
  std::array<double, 3> d_nodes{};
};

inline EikonalResidualJacobian EvalEikonalResidual(const GridMap& map, int w, int h,
                                                   const Eigen::Vector2d& normal) {
  EikonalResidualJacobian out;
  const double d00 = map.Value(w, h);
  const double d10 = map.Value(w + 1, h);
  const double d01 = map.Value(w, h + 1);
  const double gx = (d10 - d00) / map.dx();
  const double gy = (d01 - d00) / map.dy();
  out.residual = 1.0 - (normal.x() * gx + normal.y() * gy);
  out.node_ids = {map.NodeId(w, h), map.NodeId(w + 1, h), map.NodeId(w, h + 1)};
  out.d_nodes = {normal.x() / map.dx() + normal.y() / map.dy(), -normal.x() / map.dx(),
                 -normal.y() / map.dy()};
  return out;
}

/// Odometry residual r = [R(-theta_i)(t_{i+1}-t_i) - meas_t; dtheta - meas_theta]
/// with Jacobians from eq. 3.52-3.53. The angle component is wrapped so a
/// measurement across the +-pi seam does not produce a 2*pi residual.
struct OdomResidualJacobian {
  Eigen::Vector3d residual{0.0, 0.0, 0.0};
  Eigen::Matrix3d d_pose_i{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d d_pose_j{Eigen::Matrix3d::Zero()};
};

inline OdomResidualJacobian EvalOdomResidual(const Pose2& pose_i, const Pose2& pose_j,
                                             const Pose2& measurement) {
  OdomResidualJacobian out;
  const double c = std::cos(pose_i.theta);
  const double s = std::sin(pose_i.theta);
  Eigen::Matrix2d rot_neg;  // R(-theta_i)
  rot_neg << c, s, -s, c;

  const Eigen::Vector2d dt = pose_j.Translation() - pose_i.Translation();
  const Eigen::Vector2d rel_t = rot_neg * dt;
  out.residual.head<2>() = rel_t - measurement.Translation();
  out.residual[2] = WrapAngle(pose_j.theta - pose_i.theta - measurement.theta);

  Eigen::Matrix2d j_hat;
  j_hat << 0.0, -1.0, 1.0, 0.0;

  out.d_pose_i.block<2, 2>(0, 0) = -rot_neg;
  out.d_pose_i.block<2, 1>(0, 2) = -rot_neg * j_hat * dt;
  out.d_pose_i(2, 2) = -1.0;

  out.d_pose_j.block<2, 2>(0, 0) = rot_neg;
  out.d_pose_j(2, 2) = 1.0;
  return out;
}

}  // namespace sdf_slam

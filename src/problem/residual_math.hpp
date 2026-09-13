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
///
/// Outside the map domain the SDF evaluates to 0 with zero gradient, so the
/// residual -delta stays in the objective as a constant penalty. Hallucinated
/// points (delta != 0) then act as a soft barrier: a step that pushes points
/// off the map raises the cost instead of deleting their residuals, which
/// would reward the escape. `valid` is false in that case: the residual
/// carries no Jacobian entries.
/// see DEC-0006 out-of-domain-soft-barrier
struct PointResidualJacobian {
  bool valid{false};
  double residual{0.0};
  std::array<int, 4> node_ids{};
  std::array<double, 4> d_nodes{};
  Eigen::Vector3d d_pose{0.0, 0.0, 0.0};
};

/// Central-difference SDF gradient, computed per node (clamped to one-sided
/// differences at the map border) and bilinearly interpolated inside the
/// cell. Wider stencil than the bilinear-patch gradient and continuous
/// across cell borders, so pose steps do not jump at cell boundaries.
/// see DEC-0007 smooth-gradient-registration
inline Eigen::Vector2d SmoothGradient(const GridMap& map, const GridMap::CellRef& cell) {
  const auto node_gradient = [&map](int w, int h) -> Eigen::Vector2d {
    const int w_lo = std::max(w - 1, 0);
    const int w_hi = std::min(w + 1, map.nx() - 1);
    const int h_lo = std::max(h - 1, 0);
    const int h_hi = std::min(h + 1, map.ny() - 1);
    const double gx = (map.Value(w_hi, h) - map.Value(w_lo, h)) / ((w_hi - w_lo) * map.dx());
    const double gy = (map.Value(w, h_hi) - map.Value(w, h_lo)) / ((h_hi - h_lo) * map.dy());
    return {gx, gy};
  };
  const Eigen::Vector2d g00 = node_gradient(cell.w, cell.h);
  const Eigen::Vector2d g10 = node_gradient(cell.w + 1, cell.h);
  const Eigen::Vector2d g01 = node_gradient(cell.w, cell.h + 1);
  const Eigen::Vector2d g11 = node_gradient(cell.w + 1, cell.h + 1);
  const double a = cell.alpha;
  const double b = cell.beta;
  return (1.0 - a) * (1.0 - b) * g00 + a * (1.0 - b) * g10 + (1.0 - a) * b * g01 + a * b * g11;
}

/// Central finite difference of the interpolated SDF over a fixed step in
/// meters, clamped at the map border. Unlike the node-based stencil the
/// smoothing width does not change with the grid resolution.
/// see DEC-0007 smooth-gradient-registration
inline Eigen::Vector2d SmoothGradientMeters(const GridMap& map, const Eigen::Vector2d& p,
                                            double step) {
  GridMap::CellRef cell;
  const auto value = [&map, &cell](double x, double y) {
    map.LocateClamped({x, y}, cell);
    return map.Interpolate(cell);
  };
  const double gx = (value(p.x() + step, p.y()) - value(p.x() - step, p.y())) / (2.0 * step);
  const double gy = (value(p.x(), p.y() + step) - value(p.x(), p.y() - step)) / (2.0 * step);
  return {gx, gy};
}

inline PointResidualJacobian EvalPointResidual(const GridMap& map, const Pose2& pose,
                                               const Eigen::Vector2d& point_sensor, double delta,
                                               bool smooth_gradient = false,
                                               double smooth_gradient_step = 0.0) {
  PointResidualJacobian out;
  const Eigen::Vector2d point_global = pose.Apply(point_sensor);
  GridMap::CellRef cell;
  if (!map.Locate(point_global, cell)) {
    out.residual = -delta;
    return out;
  }
  out.valid = true;
  out.residual = map.Interpolate(cell) - delta;
  out.node_ids = cell.node_ids;
  out.d_nodes = cell.weights;

  Eigen::Vector2d grad;
  if (smooth_gradient && smooth_gradient_step > 0.0) {
    grad = SmoothGradientMeters(map, point_global, smooth_gradient_step);
  } else if (smooth_gradient) {
    grad = SmoothGradient(map, cell);
  } else {
    grad = map.Gradient(cell);
  }
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

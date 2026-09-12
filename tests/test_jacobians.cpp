#include <gtest/gtest.h>

#include <random>

#include "problem/residual_math.hpp"

namespace sdf_slam {
namespace {

constexpr double kEps = 1e-6;
constexpr double kTol = 1e-5;

GridMap RandomMap(std::mt19937& rng) {
  GridMap map(6, 5, {-3.0, -2.0}, {3.0, 2.0});
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  for (int i = 0; i < map.num_nodes(); ++i) {
    map.values()[i] = dist(rng);
  }
  return map;
}

/// Central finite difference of the point residual with respect to one map
/// node.
double PointResidualNodeFd(GridMap map, const Pose2& pose, const Eigen::Vector2d& p, double delta,
                           int node_id) {
  map.values()[node_id] += kEps;
  const double plus = EvalPointResidual(map, pose, p, delta).residual;
  map.values()[node_id] -= 2 * kEps;
  const double minus = EvalPointResidual(map, pose, p, delta).residual;
  return (plus - minus) / (2 * kEps);
}

TEST(Jacobians, PointResidualMatchesFiniteDifferences) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> point_dist(-1.0, 1.0);
  std::uniform_real_distribution<double> pose_dist(-0.5, 0.5);

  for (int trial = 0; trial < 20; ++trial) {
    const GridMap map = RandomMap(rng);
    const Pose2 pose{pose_dist(rng), pose_dist(rng), pose_dist(rng) * 3.0};
    const Eigen::Vector2d p(point_dist(rng), point_dist(rng));
    const double delta = point_dist(rng);

    const PointResidualJacobian eval = EvalPointResidual(map, pose, p, delta);
    if (!eval.valid) {
      continue;
    }

    for (size_t k = 0; k < 4; ++k) {
      const double fd = PointResidualNodeFd(map, pose, p, delta, eval.node_ids[k]);
      EXPECT_NEAR(eval.d_nodes[k], fd, kTol) << "node " << k << " trial " << trial;
    }

    // Pose parameters. Interpolation weights are only piecewise smooth, so a
    // point close to a cell boundary would break the finite difference; the
    // small pose perturbations here keep points interior with margin.
    const std::array<Pose2, 3> plus = {Pose2{pose.x + kEps, pose.y, pose.theta},
                                       Pose2{pose.x, pose.y + kEps, pose.theta},
                                       Pose2{pose.x, pose.y, pose.theta + kEps}};
    const std::array<Pose2, 3> minus = {Pose2{pose.x - kEps, pose.y, pose.theta},
                                        Pose2{pose.x, pose.y - kEps, pose.theta},
                                        Pose2{pose.x, pose.y, pose.theta - kEps}};
    for (int k = 0; k < 3; ++k) {
      const PointResidualJacobian eval_plus =
          EvalPointResidual(map, plus[static_cast<size_t>(k)], p, delta);
      const PointResidualJacobian eval_minus =
          EvalPointResidual(map, minus[static_cast<size_t>(k)], p, delta);
      if (!eval_plus.valid || !eval_minus.valid || eval_plus.node_ids != eval.node_ids ||
          eval_minus.node_ids != eval.node_ids) {
        continue;  // Crossed a cell boundary; the FD comparison is invalid.
      }
      const double fd = (eval_plus.residual - eval_minus.residual) / (2 * kEps);
      EXPECT_NEAR(eval.d_pose[k], fd, kTol) << "pose param " << k << " trial " << trial;
    }
  }
}

TEST(Jacobians, FrozenCellMatchesRegularEvalInsideCell) {
  std::mt19937 rng(7);
  const GridMap map = RandomMap(rng);
  const Pose2 pose{0.1, -0.2, 0.3};
  const Eigen::Vector2d p(0.4, 0.5);

  const PointResidualJacobian regular = EvalPointResidual(map, pose, p, 0.25);
  ASSERT_TRUE(regular.valid);

  GridMap::CellRef cell;
  ASSERT_TRUE(map.Locate(pose.Apply(p), cell));
  const std::array<double, 4> node_values = {
      map.values()[cell.node_ids[0]], map.values()[cell.node_ids[1]],
      map.values()[cell.node_ids[2]], map.values()[cell.node_ids[3]]};
  const PointResidualJacobian frozen =
      EvalPointResidualFrozenCell(map, cell.w, cell.h, pose, p, 0.25, node_values);

  EXPECT_NEAR(frozen.residual, regular.residual, 1e-12);
  for (size_t k = 0; k < 4; ++k) {
    EXPECT_NEAR(frozen.d_nodes[k], regular.d_nodes[k], 1e-12);
  }
  for (int k = 0; k < 3; ++k) {
    EXPECT_NEAR(frozen.d_pose[k], regular.d_pose[k], 1e-12);
  }
}

TEST(Jacobians, EikonalResidualMatchesFiniteDifferences) {
  std::mt19937 rng(43);
  GridMap map = RandomMap(rng);
  const Eigen::Vector2d normal = Eigen::Vector2d(0.6, 0.8);  // unit length

  const EikonalResidualJacobian eval = EvalEikonalResidual(map, 2, 1, normal);
  for (size_t k = 0; k < 3; ++k) {
    map.values()[eval.node_ids[k]] += kEps;
    const double plus = EvalEikonalResidual(map, 2, 1, normal).residual;
    map.values()[eval.node_ids[k]] -= 2 * kEps;
    const double minus = EvalEikonalResidual(map, 2, 1, normal).residual;
    map.values()[eval.node_ids[k]] += kEps;
    EXPECT_NEAR(eval.d_nodes[k], (plus - minus) / (2 * kEps), kTol) << "node " << k;
  }
}

TEST(Jacobians, OdomResidualMatchesFiniteDifferences) {
  const Pose2 pose_i{0.5, -1.0, 0.9};
  const Pose2 pose_j{1.5, 0.25, 1.4};
  const Pose2 meas{0.9, 1.1, 0.45};

  const OdomResidualJacobian eval = EvalOdomResidual(pose_i, pose_j, meas);

  auto perturb = [](const Pose2& pose, int param, double amount) {
    Pose2 out = pose;
    if (param == 0) {
      out.x += amount;
    } else if (param == 1) {
      out.y += amount;
    } else {
      out.theta += amount;
    }
    return out;
  };

  for (int param = 0; param < 3; ++param) {
    const Eigen::Vector3d fd_i =
        (EvalOdomResidual(perturb(pose_i, param, kEps), pose_j, meas).residual -
         EvalOdomResidual(perturb(pose_i, param, -kEps), pose_j, meas).residual) /
        (2 * kEps);
    const Eigen::Vector3d fd_j =
        (EvalOdomResidual(pose_i, perturb(pose_j, param, kEps), meas).residual -
         EvalOdomResidual(pose_i, perturb(pose_j, param, -kEps), meas).residual) /
        (2 * kEps);
    for (int r = 0; r < 3; ++r) {
      EXPECT_NEAR(eval.d_pose_i(r, param), fd_i[r], kTol) << "i param " << param << " row " << r;
      EXPECT_NEAR(eval.d_pose_j(r, param), fd_j[r], kTol) << "j param " << param << " row " << r;
    }
  }
}

}  // namespace
}  // namespace sdf_slam

#include <ceres/ceres.h>

#include <array>
#include <chrono>
#include <thread>
#include <vector>

#include "problem/residual_math.hpp"
#include "solvers/solver.hpp"

namespace sdf_slam {

namespace {

/// Scan/hallucination residual with a cell frozen at problem-build time.
/// Ceres parameter blocks are fixed per solve, so the four map nodes cannot
/// change while optimizing; a point that drifts out of its cell extrapolates
/// the cell's bilinear patch. see DEC-0004 ceres-frozen-cells
class PointCost : public ceres::SizedCostFunction<1, 1, 1, 1, 1, 3> {
 public:
  PointCost(const GridMap* map, int cell_w, int cell_h, Eigen::Vector2d point_sensor,
            double expected_sdf, double sqrt_weight)
      : map_(map),
        cell_w_(cell_w),
        cell_h_(cell_h),
        point_sensor_(std::move(point_sensor)),
        expected_sdf_(expected_sdf),
        sqrt_weight_(sqrt_weight) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const std::array<double, 4> node_values = {*parameters[0], *parameters[1], *parameters[2],
                                               *parameters[3]};
    const Pose2 pose{parameters[4][0], parameters[4][1], parameters[4][2]};
    const PointResidualJacobian eval = EvalPointResidualFrozenCell(
        *map_, cell_w_, cell_h_, pose, point_sensor_, expected_sdf_, node_values);
    residuals[0] = sqrt_weight_ * eval.residual;
    if (jacobians != nullptr) {
      for (size_t k = 0; k < 4; ++k) {
        if (jacobians[k] != nullptr) {
          jacobians[k][0] = sqrt_weight_ * eval.d_nodes[k];
        }
      }
      if (jacobians[4] != nullptr) {
        for (int k = 0; k < 3; ++k) {
          jacobians[4][k] = sqrt_weight_ * eval.d_pose[k];
        }
      }
    }
    return true;
  }

 private:
  const GridMap* map_;
  int cell_w_;
  int cell_h_;
  Eigen::Vector2d point_sensor_;
  double expected_sdf_;
  double sqrt_weight_;
};

/// Eikonal residual over the node itself and its +x / +y neighbors; the
/// Jacobian is constant (eq. 3.40-3.42).
class EikonalCost : public ceres::SizedCostFunction<1, 1, 1, 1> {
 public:
  EikonalCost(double dx, double dy, Eigen::Vector2d normal, double sqrt_weight)
      : dx_(dx), dy_(dy), normal_(std::move(normal)), sqrt_weight_(sqrt_weight) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double gx = (*parameters[1] - *parameters[0]) / dx_;
    const double gy = (*parameters[2] - *parameters[0]) / dy_;
    residuals[0] = sqrt_weight_ * (1.0 - (normal_.x() * gx + normal_.y() * gy));
    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        jacobians[0][0] = sqrt_weight_ * (normal_.x() / dx_ + normal_.y() / dy_);
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = -sqrt_weight_ * normal_.x() / dx_;
      }
      if (jacobians[2] != nullptr) {
        jacobians[2][0] = -sqrt_weight_ * normal_.y() / dy_;
      }
    }
    return true;
  }

 private:
  double dx_;
  double dy_;
  Eigen::Vector2d normal_;
  double sqrt_weight_;
};

class OdomCost : public ceres::SizedCostFunction<3, 3, 3> {
 public:
  OdomCost(Pose2 measurement, double sqrt_weight)
      : measurement_(measurement), sqrt_weight_(sqrt_weight) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const Pose2 pose_i{parameters[0][0], parameters[0][1], parameters[0][2]};
    const Pose2 pose_j{parameters[1][0], parameters[1][1], parameters[1][2]};
    const OdomResidualJacobian eval = EvalOdomResidual(pose_i, pose_j, measurement_);
    for (int r = 0; r < 3; ++r) {
      residuals[r] = sqrt_weight_ * eval.residual[r];
    }
    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac_i(jacobians[0]);
        jac_i = sqrt_weight_ * eval.d_pose_i;
      }
      if (jacobians[1] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac_j(jacobians[1]);
        jac_j = sqrt_weight_ * eval.d_pose_j;
      }
    }
    return true;
  }

 private:
  Pose2 measurement_;
  double sqrt_weight_;
};

}  // namespace

SolveResult SolveCeres(Problem& problem, const SolverOptions& options) {
  const auto start_time = std::chrono::steady_clock::now();

  SolveResult result;
  result.initial_cost = problem.Cost();

  // Parameter storage: one double per active node, one 3-array per pose.
  std::vector<double> node_values(static_cast<size_t>(problem.num_active_nodes()));
  for (int a = 0; a < problem.num_active_nodes(); ++a) {
    node_values[static_cast<size_t>(a)] = problem.map().values()[problem.ActiveNode(a)];
  }
  std::vector<std::array<double, 3>> pose_values(problem.poses().size());
  for (size_t i = 0; i < problem.poses().size(); ++i) {
    pose_values[i] = {problem.poses()[i].x, problem.poses()[i].y, problem.poses()[i].theta};
  }

  ceres::Problem ceres_problem;
  const GridMap& map = problem.map();

  auto node_param = [&](int node_id) -> double* {
    const int col = problem.NodeColumn(node_id);
    return col >= 0 ? &node_values[static_cast<size_t>(col)] : nullptr;
  };

  for (const auto& spec : problem.point_specs()) {
    const Eigen::Vector2d global =
        problem.poses()[static_cast<size_t>(spec.frame)].Apply(spec.point_sensor);
    GridMap::CellRef cell;
    if (!map.Locate(global, cell)) {
      continue;
    }
    std::array<double*, 4> nodes{};
    bool all_active = true;
    for (size_t k = 0; k < 4; ++k) {
      nodes[k] = node_param(cell.node_ids[k]);
      all_active = all_active && nodes[k] != nullptr;
    }
    if (!all_active) {
      continue;
    }
    ceres::LossFunction* loss =
        problem.huber_delta() > 0.0 ? new ceres::HuberLoss(problem.huber_delta()) : nullptr;
    ceres_problem.AddResidualBlock(
        new PointCost(&map, cell.w, cell.h, spec.point_sensor, spec.expected_sdf, spec.sqrt_weight),
        loss, nodes[0], nodes[1], nodes[2], nodes[3],
        pose_values[static_cast<size_t>(spec.frame)].data());
  }

  for (const auto& spec : problem.eikonal_specs()) {
    double* n0 = node_param(map.NodeId(spec.w, spec.h));
    double* n1 = node_param(map.NodeId(spec.w + 1, spec.h));
    double* n2 = node_param(map.NodeId(spec.w, spec.h + 1));
    if (n0 == nullptr || n1 == nullptr || n2 == nullptr) {
      continue;
    }
    ceres_problem.AddResidualBlock(
        new EikonalCost(map.dx(), map.dy(), spec.normal, spec.sqrt_weight), nullptr, n0, n1, n2);
  }

  for (const auto& spec : problem.odom_specs()) {
    ceres_problem.AddResidualBlock(new OdomCost(spec.measurement, spec.sqrt_weight), nullptr,
                                   pose_values[static_cast<size_t>(spec.frame_i)].data(),
                                   pose_values[static_cast<size_t>(spec.frame_i) + 1].data());
  }

  if (ceres_problem.HasParameterBlock(pose_values[0].data())) {
    ceres_problem.SetParameterBlockConstant(pose_values[0].data());
  }

  ceres::Solver::Options ceres_options;
  ceres_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  ceres_options.max_num_iterations = options.max_iterations;
  ceres_options.num_threads = options.num_threads > 0
                                  ? options.num_threads
                                  : static_cast<int>(std::thread::hardware_concurrency());
  ceres_options.initial_trust_region_radius = 1.0 / std::max(options.lambda_init, 1e-12);
  ceres_options.parameter_tolerance = options.step_tolerance;
  ceres_options.logging_type = ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options, &ceres_problem, &summary);

  // Copy the solution back.
  for (int a = 0; a < problem.num_active_nodes(); ++a) {
    problem.map().values()[problem.ActiveNode(a)] = node_values[static_cast<size_t>(a)];
  }
  for (size_t i = 0; i < problem.poses().size(); ++i) {
    problem.poses()[i] = {pose_values[i][0], pose_values[i][1], WrapAngle(pose_values[i][2])};
  }

  result.converged = summary.termination_type == ceres::CONVERGENCE;
  result.iterations = static_cast<int>(summary.iterations.size());
  result.final_cost = problem.Cost();
  result.duration_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  return result;
}

}  // namespace sdf_slam

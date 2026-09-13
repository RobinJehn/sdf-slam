#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "problem/problem.hpp"
#include "solvers/solver.hpp"

namespace sdf_slam {
namespace {

/// Two perpendicular walls (x = 0 with y in [0, 10], y = 0 with x in [0, 10])
/// observed from a pose. The corner constrains both translation axes and the
/// rotation.
Scan CornerScan(const Pose2& pose) {
  Scan scan;
  const Eigen::Matrix2d rot_inv = pose.Rotation().transpose();
  auto add_world_point = [&](const Eigen::Vector2d& world) {
    scan.points.emplace_back(rot_inv * (world - pose.Translation()));
  };
  for (double s = 0.25; s <= 10.0; s += 0.25) {
    add_world_point({0.0, s});
    add_world_point({s, 0.0});
  }
  return scan;
}

struct E2eSetup {
  std::vector<Scan> scans;
  std::vector<Pose2> poses_init;
  std::vector<Pose2> odometry;
  Pose2 gt_pose_1;
};

E2eSetup MakeSetup() {
  E2eSetup setup;
  const Pose2 pose_0{4.0, 4.0, 0.2};
  setup.gt_pose_1 = {5.0, 5.0, -0.1};
  setup.scans = {CornerScan(pose_0), CornerScan(setup.gt_pose_1)};
  // Frame 1 starts away from the truth; odometry measures the true relative
  // motion, scan and map residuals must agree with it.
  setup.poses_init = {pose_0, {5.4, 4.7, 0.05}};
  setup.odometry = {pose_0.RelativeTo(setup.gt_pose_1)};
  return setup;
}

ProblemOptions MakeProblemOptions() {
  ProblemOptions options;
  options.hallucination.points_per_scan_point = 4;
  options.hallucination.step_size = 0.2;
  options.normals.k_neighbors = 8;
  // The dissertation-baseline method keeps the convergence bounds tight; the
  // ablation of methods lives in configs, not in this machinery test.
  options.normals.method = NormalMethod::kPca;
  return options;
}

GridMap MakeE2eMap() { return {40, 40, {-2.0, -2.0}, {12.0, 12.0}, 0.0}; }

void ExpectRecovered(const Problem& problem, const E2eSetup& setup, double initial_cost,
                     double final_cost) {
  // The optimization must reduce the cost substantially...
  EXPECT_LT(final_cost, 0.05 * initial_cost);
  // ...pull frame 1 toward the ground truth...
  const Pose2& estimated = problem.poses()[1];
  const double translation_error = (estimated.Translation() - setup.gt_pose_1.Translation()).norm();
  const double init_error =
      (setup.poses_init[1].Translation() - setup.gt_pose_1.Translation()).norm();
  EXPECT_LT(translation_error, 0.1);
  EXPECT_LT(translation_error, init_error);
  EXPECT_LT(std::abs(WrapAngle(estimated.theta - setup.gt_pose_1.theta)), 0.05);

  // ...and drive the SDF toward zero on the observed surfaces.
  double surface_abs_sum = 0.0;
  int surface_count = 0;
  for (const auto& p : setup.scans[0].points) {
    GridMap::CellRef cell;
    if (problem.map().Locate(problem.poses()[0].Apply(p), cell)) {
      surface_abs_sum += std::abs(problem.map().Interpolate(cell));
      ++surface_count;
    }
  }
  ASSERT_GT(surface_count, 0);
  EXPECT_LT(surface_abs_sum / surface_count, 0.1);
}

TEST(EndToEnd, LmSolverRecoversPerturbedPose) {
  const E2eSetup setup = MakeSetup();
  Problem problem(MakeE2eMap(), setup.poses_init, setup.scans, setup.odometry,
                  MakeProblemOptions());

  SolverOptions options;
  options.backend = SolverBackend::kLm;
  options.max_iterations = 40;
  const SolveResult result = Solve(problem, options);

  ExpectRecovered(problem, setup, result.initial_cost, result.final_cost);
}

TEST(EndToEnd, CeresSolverRecoversPerturbedPose) {
  const E2eSetup setup = MakeSetup();
  Problem problem(MakeE2eMap(), setup.poses_init, setup.scans, setup.odometry,
                  MakeProblemOptions());

  SolverOptions options;
  options.backend = SolverBackend::kCeres;
  options.max_iterations = 60;
  const SolveResult result = Solve(problem, options);

  ExpectRecovered(problem, setup, result.initial_cost, result.final_cost);
}

TEST(EndToEnd, TrustRegionLmRecoversPerturbedPose) {
  const E2eSetup setup = MakeSetup();
  Problem problem(MakeE2eMap(), setup.poses_init, setup.scans, setup.odometry,
                  MakeProblemOptions());

  SolverOptions options;
  options.backend = SolverBackend::kLm;
  options.trust_region = true;
  options.max_iterations = 40;
  const SolveResult result = Solve(problem, options);

  ExpectRecovered(problem, setup, result.initial_cost, result.final_cost);
  // The gain-ratio strategy must never keep a step that raised the cost.
  for (size_t i = 1; i < result.cost_history.size(); ++i) {
    EXPECT_LE(result.cost_history[i], result.cost_history[i - 1] + 1e-9);
  }
}

TEST(EndToEnd, HuberLossStillRecoversPerturbedPose) {
  const E2eSetup setup = MakeSetup();
  ProblemOptions options = MakeProblemOptions();
  options.huber_delta = 0.5;
  Problem problem(MakeE2eMap(), setup.poses_init, setup.scans, setup.odometry, options);

  SolverOptions solver_options;
  solver_options.backend = SolverBackend::kLm;
  solver_options.max_iterations = 40;
  const SolveResult result = Solve(problem, solver_options);

  ExpectRecovered(problem, setup, result.initial_cost, result.final_cost);
}

TEST(EndToEnd, ActiveRegionShrinksStateAndStillConverges) {
  const E2eSetup setup = MakeSetup();

  ProblemOptions dense_options = MakeProblemOptions();
  const Problem dense(MakeE2eMap(), setup.poses_init, setup.scans, setup.odometry, dense_options);

  ProblemOptions active_options = MakeProblemOptions();
  active_options.active_region = true;
  active_options.active_margin = 2;
  Problem active(MakeE2eMap(), setup.poses_init, setup.scans, setup.odometry, active_options);

  EXPECT_LT(active.num_active_nodes(), dense.num_active_nodes());

  SolverOptions options;
  options.backend = SolverBackend::kLm;
  options.max_iterations = 40;
  const SolveResult result = Solve(active, options);
  ExpectRecovered(active, setup, result.initial_cost, result.final_cost);
}

}  // namespace
}  // namespace sdf_slam

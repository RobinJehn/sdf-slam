#pragma once

#include <cstdint>
#include <vector>

#include "problem/problem.hpp"

namespace sdf_slam {

enum class SolverBackend : std::uint8_t {
  /// Hand-rolled Levenberg-Marquardt with Eigen sparse Cholesky
  /// (dissertation algorithm 1).
  kLm,
  /// Ceres with the same analytic residuals. see DEC-0001 dual-solver-backends
  kCeres,
};

struct SolverOptions {
  SolverBackend backend{SolverBackend::kLm};
  int max_iterations{100};
  /// Stop when the step norm drops below this tolerance.
  double step_tolerance{1e-6};
  double lambda_init{1.0};
  /// Multiplicative damping adjustment (dissertation algorithm 1). A factor
  /// of 1 keeps the damping fixed, which is the dissertation's table 4.1
  /// setting.
  double lambda_factor{1.0};
  /// When true, a step that increases the cost is reverted (classic LM).
  /// Algorithm 1 never reverts, so the default follows the dissertation.
  bool reject_worse_steps{false};
  int num_threads{0};  // 0 = hardware concurrency
};

struct SolveResult {
  bool converged{false};
  int iterations{0};
  double initial_cost{0.0};
  double final_cost{0.0};
  double duration_seconds{0.0};
  std::vector<double> cost_history{};
};

/// Minimizes the problem in place with the selected backend.
SolveResult Solve(Problem& problem, const SolverOptions& options);

SolveResult SolveLm(Problem& problem, const SolverOptions& options);
SolveResult SolveCeres(Problem& problem, const SolverOptions& options);

}  // namespace sdf_slam

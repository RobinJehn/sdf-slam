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

enum class LinearSolver : std::uint8_t {
  /// Eigen SimplicialLDLT: single-threaded, bit-reproducible with earlier
  /// results.
  kEigen,
  /// CHOLMOD supernodal LLT (SuiteSparse): multithreaded BLAS kernels, much
  /// faster on grid-sized normal matrices. The elimination ordering differs
  /// from Eigen's, so steps differ in floating-point rounding; results are
  /// metric-equivalent but not bit-identical to the Eigen path.
  kCholmod,
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
  /// When true, the damping term is lambda * diag(J^T J) (Marquardt scaling,
  /// as in Eigen's LevenbergMarquardt) instead of lambda * I. Scales the step
  /// per parameter, which matters when map values and pose angles mix.
  bool marquardt_scaling{false};
  /// When true, lambda follows Nielsen's gain-ratio trust-region strategy:
  /// rho = (actual cost reduction) / (reduction the linear model predicts).
  /// Steps with rho <= 0 are reverted and lambda grows geometrically; good
  /// steps shrink lambda smoothly. Overrides lambda_factor and
  /// reject_worse_steps.
  bool trust_region{false};
  int num_threads{0};  // 0 = hardware concurrency
  /// Sparse factorization used for the normal equations (LM backend only).
  LinearSolver linear_solver{LinearSolver::kEigen};
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

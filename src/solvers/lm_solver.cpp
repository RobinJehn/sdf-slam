#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

#include "problem/residual_math.hpp"
#include "solvers/solver.hpp"

namespace sdf_slam {

namespace {

using Triplet = Eigen::Triplet<double>;

struct AssemblyChunk {
  std::vector<Triplet> triplets;
  std::vector<std::pair<int, double>> residuals;  // (row, value)
};

/// Assembles the point-residual rows [begin, end). Rows are indexed by spec
/// position, so chunks are independent and run in parallel.
void AssemblePointRows(const Problem& problem, size_t begin, size_t end, AssemblyChunk& chunk) {
  const GridMap& map = problem.map();
  const std::vector<Pose2>& poses = problem.poses();
  for (size_t s = begin; s < end; ++s) {
    const PointSpec& spec = problem.point_specs()[s];
    const PointResidualJacobian eval =
        EvalPointResidual(map, poses[static_cast<size_t>(spec.frame)], spec.point_sensor,
                          spec.expected_sdf, problem.smooth_gradient());
    const int row = static_cast<int>(s);
    const double weighted = spec.sqrt_weight * eval.residual;
    // IRLS: sqrt of the Huber weight scales the row so the Gauss-Newton step
    // minimizes the robust cost around the current residual.
    const double sw = std::sqrt(problem.HuberWeight(weighted));
    chunk.residuals.emplace_back(row, sw * weighted);
    if (!eval.valid) {
      continue;  // Point outside the map: constant residual, no Jacobian row.
    }
    const double scale = sw * spec.sqrt_weight;
    for (size_t k = 0; k < 4; ++k) {
      const int col = problem.NodeColumn(eval.node_ids[k]);
      if (col >= 0) {
        chunk.triplets.emplace_back(row, col, scale * eval.d_nodes[k]);
      }
    }
    if (spec.frame >= 1) {
      const int pose_col = problem.PoseColumn(spec.frame);
      for (int k = 0; k < 3; ++k) {
        chunk.triplets.emplace_back(row, pose_col + k, scale * eval.d_pose[k]);
      }
    }
  }
}

}  // namespace

SolveResult SolveLm(Problem& problem, const SolverOptions& options) {
  const auto start_time = std::chrono::steady_clock::now();

  SolveResult result;
  result.initial_cost = problem.Cost();
  result.cost_history.push_back(result.initial_cost);

  const int num_params = problem.num_parameters();
  const int num_point_rows = static_cast<int>(problem.point_specs().size());
  const int num_eik_rows = static_cast<int>(problem.eikonal_specs().size());
  const int num_rows = problem.num_residuals();

  double lambda = options.lambda_init;
  double previous_cost = result.initial_cost;
  double nielsen_nu = 2.0;  // growth factor for rejected steps (Nielsen 1999)

  const unsigned hardware = std::thread::hardware_concurrency();
  const size_t num_threads =
      options.num_threads > 0 ? static_cast<size_t>(options.num_threads) : std::max(1U, hardware);

  for (int iter = 0; iter < options.max_iterations; ++iter) {
    // Residual vector and Jacobian triplets. Point rows are assembled in
    // parallel chunks; Eikonal and odometry rows are cheap and stay serial.
    std::vector<AssemblyChunk> chunks(num_threads);
    {
      std::vector<std::thread> workers;
      const size_t total = problem.point_specs().size();
      const size_t chunk_size = (total + num_threads - 1) / num_threads;
      for (size_t t = 0; t < num_threads; ++t) {
        const size_t begin = t * chunk_size;
        const size_t end = std::min(total, begin + chunk_size);
        if (begin >= end) {
          break;
        }
        workers.emplace_back(AssemblePointRows, std::cref(problem), begin, end,
                             std::ref(chunks[t]));
      }
      for (auto& worker : workers) {
        worker.join();
      }
    }

    AssemblyChunk tail;
    for (int e = 0; e < num_eik_rows; ++e) {
      const EikonalSpec& spec = problem.eikonal_specs()[static_cast<size_t>(e)];
      const EikonalResidualJacobian eval =
          EvalEikonalResidual(problem.map(), spec.w, spec.h, spec.normal);
      const int row = num_point_rows + e;
      tail.residuals.emplace_back(row, spec.sqrt_weight * eval.residual);
      for (size_t k = 0; k < 3; ++k) {
        const int col = problem.NodeColumn(eval.node_ids[k]);
        if (col >= 0) {
          tail.triplets.emplace_back(row, col, spec.sqrt_weight * eval.d_nodes[k]);
        }
      }
    }
    for (size_t o = 0; o < problem.odom_specs().size(); ++o) {
      const OdomSpec& spec = problem.odom_specs()[o];
      const OdomResidualJacobian eval = EvalOdomResidual(
          problem.poses()[static_cast<size_t>(spec.frame_i)],
          problem.poses()[static_cast<size_t>(spec.frame_i) + 1], spec.measurement);
      const int row = num_point_rows + num_eik_rows + 3 * static_cast<int>(o);
      for (int r = 0; r < 3; ++r) {
        tail.residuals.emplace_back(row + r, spec.sqrt_weight * eval.residual[r]);
        if (spec.frame_i >= 1) {
          const int col_i = problem.PoseColumn(spec.frame_i);
          for (int c = 0; c < 3; ++c) {
            tail.triplets.emplace_back(row + r, col_i + c, spec.sqrt_weight * eval.d_pose_i(r, c));
          }
        }
        const int col_j = problem.PoseColumn(spec.frame_i + 1);
        for (int c = 0; c < 3; ++c) {
          tail.triplets.emplace_back(row + r, col_j + c, spec.sqrt_weight * eval.d_pose_j(r, c));
        }
      }
    }
    chunks.push_back(std::move(tail));

    size_t total_triplets = 0;
    for (const auto& chunk : chunks) {
      total_triplets += chunk.triplets.size();
    }
    std::vector<Triplet> triplets;
    triplets.reserve(total_triplets);
    Eigen::VectorXd residuals = Eigen::VectorXd::Zero(num_rows);
    for (const auto& chunk : chunks) {
      triplets.insert(triplets.end(), chunk.triplets.begin(), chunk.triplets.end());
      for (const auto& [row, value] : chunk.residuals) {
        residuals[row] = value;
      }
    }

    Eigen::SparseMatrix<double> jacobian(num_rows, num_params);
    jacobian.setFromTriplets(triplets.begin(), triplets.end());

    // Normal equations (J^T J + lambda I) dx = -J^T r (eq. 3.28), solved with
    // a sparse Cholesky factorization (section 3.3.5).
    Eigen::SparseMatrix<double> normal_matrix = jacobian.transpose() * jacobian;
    for (int d = 0; d < num_params; ++d) {
      if (options.marquardt_scaling) {
        // Guard against zero diagonal entries (parameters with no residual
        // this iteration) so the system stays positive definite.
        const double diag = std::max(normal_matrix.coeff(d, d), 1e-12);
        normal_matrix.coeffRef(d, d) += lambda * diag;
      } else {
        normal_matrix.coeffRef(d, d) += lambda;
      }
    }
    const Eigen::VectorXd rhs = -jacobian.transpose() * residuals;

    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> cholesky(normal_matrix);
    if (cholesky.info() != Eigen::Success) {
      throw std::runtime_error("sparse Cholesky factorization failed");
    }
    const Eigen::VectorXd step = cholesky.solve(rhs);

    // Snapshot for optional revert.
    Eigen::VectorXd map_backup;
    std::vector<Pose2> pose_backup;
    if (options.reject_worse_steps || options.trust_region) {
      map_backup = problem.map().values();
      pose_backup = problem.poses();
    }

    problem.ApplyStep(step);
    const double cost = problem.Cost();
    result.iterations = iter + 1;

    if (step.norm() < options.step_tolerance) {
      result.cost_history.push_back(cost);
      result.converged = true;
      break;
    }

    if (options.trust_region) {
      // Gain ratio: actual reduction over the reduction the linearization
      // predicts (Madsen et al. 2004, eq. 2.20, with F = 0.5 r^T r). Cost()
      // returns r^T r, so the 1/2 factors cancel:
      // rho = (prev - cost) / (step^T (lambda * step + rhs)), rhs = -J^T r.
      const double predicted = step.dot(lambda * step + rhs);
      const double rho = predicted > 0.0 ? (previous_cost - cost) / predicted : -1.0;
      if (rho > 0.0) {
        previous_cost = cost;
        const double shrink = 1.0 - std::pow(2.0 * rho - 1.0, 3);
        // Floor keeps the normal matrix positive definite: grid nodes with no
        // residual row have a zero J^T J diagonal and only lambda regularizes
        // them.
        lambda = std::max(lambda * std::max(1.0 / 3.0, shrink), 1e-7);
        nielsen_nu = 2.0;
      } else {
        problem.map().values() = map_backup;
        problem.poses() = pose_backup;
        lambda *= nielsen_nu;
        nielsen_nu *= 2.0;
      }
    } else if (cost < previous_cost) {
      lambda /= options.lambda_factor;
      previous_cost = cost;
    } else {
      lambda *= options.lambda_factor;
      if (options.reject_worse_steps) {
        problem.map().values() = map_backup;
        problem.poses() = pose_backup;
      } else {
        previous_cost = cost;
      }
    }
    result.cost_history.push_back(problem.Cost());
  }

  result.final_cost = problem.Cost();
  result.duration_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  return result;
}

SolveResult Solve(Problem& problem, const SolverOptions& options) {
  switch (options.backend) {
    case SolverBackend::kLm:
      return SolveLm(problem, options);
    case SolverBackend::kCeres:
      return SolveCeres(problem, options);
  }
  throw std::logic_error("unknown solver backend");
}

}  // namespace sdf_slam

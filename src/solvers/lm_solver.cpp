#include <Eigen/CholmodSupport>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "problem/residual_math.hpp"
#include "solvers/parallel_ldlt.hpp"
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
    const PointResidualJacobian eval = EvalPointResidual(
        map, poses[static_cast<size_t>(spec.frame)], spec.point_sensor, spec.expected_sdf,
        problem.smooth_gradient(), problem.smooth_gradient_step());
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

/// Assembles the Eikonal-residual rows [begin, end). Rows are indexed by spec
/// position, so chunks are independent and run in parallel. Triplet order
/// across chunks does not affect the assembled Jacobian: no two triplets
/// share a coordinate, so setFromTriplets only places values.
void AssembleEikonalRows(const Problem& problem, int num_point_rows, size_t begin, size_t end,
                         AssemblyChunk& chunk) {
  for (size_t e = begin; e < end; ++e) {
    const EikonalSpec& spec = problem.eikonal_specs()[e];
    const EikonalResidualJacobian eval =
        EvalEikonalResidual(problem.map(), spec.w, spec.h, spec.normal);
    const int row = num_point_rows + static_cast<int>(e);
    chunk.residuals.emplace_back(row, spec.sqrt_weight * eval.residual);
    for (size_t k = 0; k < 3; ++k) {
      const int col = problem.NodeColumn(eval.node_ids[k]);
      if (col >= 0) {
        chunk.triplets.emplace_back(row, col, spec.sqrt_weight * eval.d_nodes[k]);
      }
    }
  }
}

/// Runs fn(begin, end) over [0, n) split into per-thread ranges.
void ParallelRanges(size_t n, size_t num_threads, const std::function<void(size_t, size_t)>& fn) {
  const size_t threads = std::min(n, std::max<size_t>(1, num_threads));
  if (threads <= 1) {
    fn(0, n);
    return;
  }
  const size_t chunk = (n + threads - 1) / threads;
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (size_t t = 0; t < threads; ++t) {
    const size_t begin = t * chunk;
    const size_t end = std::min(n, begin + chunk);
    if (begin >= end) {
      break;
    }
    workers.emplace_back([begin, end, &fn] { fn(begin, end); });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

/// J^T J with one thread per column range. Each output column accumulates in
/// the same order as Eigen's serial conservative sparse product (ascending k
/// within the column, ascending i within row k), so every entry is bitwise
/// equal to the serial result and independent of the thread count. Inner
/// indices are emitted sorted.
/// see DEC-0008 bitwise-exact-performance-work
Eigen::SparseMatrix<double> ParallelAtA(const Eigen::SparseMatrix<double>& jt,
                                        const Eigen::SparseMatrix<double>& jacobian,
                                        size_t num_threads) {
  const Eigen::Index cols = jacobian.cols();
  std::vector<std::vector<int>> col_indices(static_cast<size_t>(cols));
  std::vector<std::vector<double>> col_values(static_cast<size_t>(cols));

  ParallelRanges(static_cast<size_t>(cols), num_threads, [&](size_t begin, size_t end) {
    const Eigen::Index rows = jt.rows();
    std::vector<char> mask(static_cast<size_t>(rows), 0);
    std::vector<double> values(static_cast<size_t>(rows), 0.0);
    std::vector<int> indices;
    for (size_t j = begin; j < end; ++j) {
      indices.clear();
      for (Eigen::SparseMatrix<double>::InnerIterator rhs_it(jacobian,
                                                             static_cast<Eigen::Index>(j));
           rhs_it; ++rhs_it) {
        const double y = rhs_it.value();
        const Eigen::Index k = rhs_it.index();
        for (Eigen::SparseMatrix<double>::InnerIterator lhs_it(jt, k); lhs_it; ++lhs_it) {
          const auto i = static_cast<size_t>(lhs_it.index());
          const double x = lhs_it.value();
          if (mask[i] == 0) {
            mask[i] = 1;
            values[i] = x * y;
            indices.push_back(static_cast<int>(i));
          } else {
            values[i] += x * y;
          }
        }
      }
      std::ranges::sort(indices);
      auto& out_idx = col_indices[j];
      auto& out_val = col_values[j];
      out_idx.reserve(indices.size());
      out_val.reserve(indices.size());
      for (const int i : indices) {
        out_idx.push_back(i);
        out_val.push_back(values[static_cast<size_t>(i)]);
        mask[static_cast<size_t>(i)] = 0;
      }
    }
  });

  Eigen::SparseMatrix<double> result(jt.rows(), cols);
  Eigen::VectorXi sizes(cols);
  for (Eigen::Index j = 0; j < cols; ++j) {
    sizes[j] = static_cast<int>(col_indices[static_cast<size_t>(j)].size());
  }
  result.reserve(sizes);
  for (Eigen::Index j = 0; j < cols; ++j) {
    const auto& idx = col_indices[static_cast<size_t>(j)];
    const auto& val = col_values[static_cast<size_t>(j)];
    for (size_t e = 0; e < idx.size(); ++e) {
      result.insert(idx[e], j) = val[e];
    }
  }
  result.makeCompressed();
  return result;
}

/// -J^T r with one thread per column range: entry j is the dot product of
/// column j of J with r. The loop mirrors Eigen's sparse-times-dense kernel,
/// which alternates entries between two accumulators; with the same
/// association the result is bitwise identical to the serial expression.
Eigen::VectorXd ParallelNegAtb(const Eigen::SparseMatrix<double>& jacobian,
                               const Eigen::VectorXd& residuals, size_t num_threads) {
  Eigen::VectorXd rhs(jacobian.cols());
  ParallelRanges(static_cast<size_t>(jacobian.cols()), num_threads, [&](size_t begin, size_t end) {
    for (size_t j = begin; j < end; ++j) {
      double tmp_a = 0.0;
      double tmp_b = 0.0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(jacobian, static_cast<Eigen::Index>(j));
           it; ++it) {
        tmp_a += it.value() * residuals[it.index()];
        ++it;
        if (it) {
          tmp_b += it.value() * residuals[it.index()];
        }
      }
      rhs[static_cast<Eigen::Index>(j)] = -(tmp_a + tmp_b);
    }
  });
  return rhs;
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

  // The sparsity pattern of J^T J changes only when a point crosses a cell
  // boundary between iterations, so the symbolic analysis (elimination
  // ordering) is reused while the pattern stays identical. Identical pattern
  // gives an identical ordering, so the factorization is bitwise equal to a
  // fresh analyze-and-factorize.
  ParallelSimplicialLdlt cholesky;
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> cholmod;
  const bool use_cholmod = options.linear_solver == LinearSolver::kCholmod;
  std::vector<int> pattern_outer;
  std::vector<int> pattern_inner;

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

    {
      std::vector<AssemblyChunk> eik_chunks(num_threads);
      std::vector<std::thread> workers;
      const auto total = static_cast<size_t>(num_eik_rows);
      const size_t chunk_size = (total + num_threads - 1) / num_threads;
      for (size_t t = 0; t < num_threads; ++t) {
        const size_t begin = t * chunk_size;
        const size_t end = std::min(total, begin + chunk_size);
        if (begin >= end) {
          break;
        }
        workers.emplace_back(AssembleEikonalRows, std::cref(problem), num_point_rows, begin, end,
                             std::ref(eik_chunks[t]));
      }
      for (auto& worker : workers) {
        worker.join();
      }
      for (auto& chunk : eik_chunks) {
        chunks.push_back(std::move(chunk));
      }
    }
    AssemblyChunk tail;
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
    // a sparse Cholesky factorization (section 3.3.5). The product runs on
    // parallel column ranges; every entry keeps the serial accumulation
    // order, so the result is bitwise equal to Eigen's serial product.
    const Eigen::SparseMatrix<double> jt = jacobian.transpose();
    Eigen::SparseMatrix<double> normal_matrix = ParallelAtA(jt, jacobian, num_threads);
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
    const Eigen::VectorXd rhs = ParallelNegAtb(jacobian, residuals, num_threads);

    normal_matrix.makeCompressed();
    const int outer_size = static_cast<int>(normal_matrix.outerSize()) + 1;
    const int nnz = static_cast<int>(normal_matrix.nonZeros());
    const bool same_pattern =
        std::cmp_equal(pattern_outer.size(), outer_size) &&
        std::cmp_equal(pattern_inner.size(), nnz) &&
        std::equal(pattern_outer.begin(), pattern_outer.end(), normal_matrix.outerIndexPtr()) &&
        std::equal(pattern_inner.begin(), pattern_inner.end(), normal_matrix.innerIndexPtr());
    if (!same_pattern) {
      if (use_cholmod) {
        cholmod.analyzePattern(normal_matrix);
      } else {
        cholesky.AnalyzePatternWithSchedule(normal_matrix, num_threads);
      }
      pattern_outer.assign(normal_matrix.outerIndexPtr(),
                           normal_matrix.outerIndexPtr() + outer_size);
      pattern_inner.assign(normal_matrix.innerIndexPtr(), normal_matrix.innerIndexPtr() + nnz);
    }
    Eigen::VectorXd step;
    if (use_cholmod) {
      cholmod.factorize(normal_matrix);
      if (cholmod.info() != Eigen::Success) {
        throw std::runtime_error("sparse Cholesky factorization failed");
      }
      step = cholmod.solve(rhs);
    } else {
      cholesky.FactorizeParallel(normal_matrix);
      if (cholesky.info() != Eigen::Success) {
        throw std::runtime_error("sparse Cholesky factorization failed");
      }
      step = cholesky.solve(rhs);
    }

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
      result.final_cost = cost;
      result.converged = true;
      break;
    }

    // Cost of the state the iteration ends in. A reverted step restores the
    // previous state bitwise, so its cost is the previous accepted cost; no
    // recomputation is needed.
    double current_cost = cost;
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
        current_cost = previous_cost;
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
        current_cost = previous_cost;
      } else {
        previous_cost = cost;
      }
    }
    result.cost_history.push_back(current_cost);
    result.final_cost = current_cost;
  }

  if (result.iterations == 0) {
    result.final_cost = result.initial_cost;
  }
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

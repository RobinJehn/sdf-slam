#include <gtest/gtest.h>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <bit>
#include <cstdint>
#include <random>
#include <vector>

#include "solvers/parallel_ldlt.hpp"

namespace sdf_slam {
namespace {

/// True when every element matches bit for bit. Chaos sensitivity of the
/// incremental recipe makes plain double equality insufficient: -0.0 == 0.0
/// yet the two diverge downstream.
bool BitwiseEqual(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    if (std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i])) {
      return false;
    }
  }
  return true;
}

/// Random SPD matrix with a banded, grid-like pattern similar to the SLAM
/// normal equations.
Eigen::SparseMatrix<double> RandomSpd(int n, int width, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(n) * 5);
  for (int i = 0; i < n; ++i) {
    trips.emplace_back(i, i, 30.0 + std::abs(dist(rng)));
    if (i + 1 < n) {
      const double v = dist(rng);
      trips.emplace_back(i, i + 1, v);
      trips.emplace_back(i + 1, i, v);
    }
    if (i + width < n) {
      const double v = dist(rng);
      trips.emplace_back(i, i + width, v);
      trips.emplace_back(i + width, i, v);
    }
  }
  Eigen::SparseMatrix<double> a(n, n);
  a.setFromTriplets(trips.begin(), trips.end());
  a.makeCompressed();
  return a;
}

TEST(ParallelLdlt, BitwiseEqualToEigenAcrossThreadCounts) {
  for (unsigned seed : {11U, 12U, 13U}) {
    const int n = 3000 + static_cast<int>(seed) * 500;
    const Eigen::SparseMatrix<double> a = RandomSpd(n, 57, seed);
    const Eigen::VectorXd b = Eigen::VectorXd::Random(n);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> reference;
    reference.analyzePattern(a);
    reference.factorize(a);
    ASSERT_EQ(reference.info(), Eigen::Success);
    const Eigen::VectorXd x_ref = reference.solve(b);

    for (const std::size_t threads : {std::size_t{1}, std::size_t{4}, std::size_t{8}}) {
      ParallelSimplicialLdlt parallel;
      parallel.AnalyzePatternWithSchedule(a, threads);
      parallel.FactorizeParallel(a);
      ASSERT_EQ(parallel.info(), Eigen::Success) << "threads " << threads;
      const Eigen::VectorXd x_par = parallel.solve(b);
      EXPECT_TRUE(BitwiseEqual(x_ref, x_par)) << "seed " << seed << " threads " << threads;
    }
  }
}

TEST(ParallelLdlt, RefactorizeWithNewValuesStaysBitwise) {
  const int n = 4000;
  const Eigen::SparseMatrix<double> a = RandomSpd(n, 63, 42);
  Eigen::SparseMatrix<double> a2 = a;
  for (int i = 0; i < n; ++i) {
    a2.coeffRef(i, i) += 2.5;
  }
  const Eigen::VectorXd b = Eigen::VectorXd::Random(n);

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> reference;
  reference.analyzePattern(a);
  reference.factorize(a);
  reference.factorize(a2);
  const Eigen::VectorXd x_ref = reference.solve(b);

  ParallelSimplicialLdlt parallel;
  parallel.AnalyzePatternWithSchedule(a, 6);
  parallel.FactorizeParallel(a);
  parallel.FactorizeParallel(a2);
  ASSERT_EQ(parallel.info(), Eigen::Success);
  const Eigen::VectorXd x_par = parallel.solve(b);
  EXPECT_TRUE(BitwiseEqual(x_ref, x_par));
}

TEST(ParallelLdlt, ReportsNumericalIssueOnSingularMatrix) {
  const int n = 600;
  // Rank-deficient: two identical rows/columns after elimination.
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    trips.emplace_back(i, i, i == n / 2 ? 0.0 : 2.0);
  }
  Eigen::SparseMatrix<double> a(n, n);
  a.setFromTriplets(trips.begin(), trips.end());

  ParallelSimplicialLdlt parallel;
  parallel.AnalyzePatternWithSchedule(a, 4);
  parallel.FactorizeParallel(a);
  EXPECT_EQ(parallel.info(), Eigen::NumericalIssue);
}

}  // namespace
}  // namespace sdf_slam

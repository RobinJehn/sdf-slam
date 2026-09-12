#include <benchmark/benchmark.h>

#include <filesystem>

#include "core/scan.hpp"
#include "problem/problem.hpp"
#include "solvers/solver.hpp"

namespace {

using sdf_slam::Dataset;
using sdf_slam::GridMap;
using sdf_slam::Pose2;
using sdf_slam::Problem;
using sdf_slam::ProblemOptions;
using sdf_slam::SolverBackend;
using sdf_slam::SolverOptions;

const Dataset& Simu10() {
  static const Dataset dataset =
      sdf_slam::LoadDataset(std::filesystem::path(SDF_SLAM_DATA_DIR) / "simu_10");
  return dataset;
}

std::vector<Pose2> RelativeOdometry(const std::vector<Pose2>& poses) {
  std::vector<Pose2> odometry;
  for (size_t i = 0; i + 1 < poses.size(); ++i) {
    odometry.push_back(poses[i].RelativeTo(poses[i + 1]));
  }
  return odometry;
}

Problem MakeProblem(bool active_region) {
  const Dataset& dataset = Simu10();
  ProblemOptions options;
  options.active_region = active_region;
  GridMap map(100, 100, {-25.0, -25.0}, {25.0, 25.0});
  return {std::move(map), dataset.poses, dataset.scans, RelativeOdometry(dataset.poses), options};
}

void RunBackend(benchmark::State& state, SolverBackend backend, bool active_region) {
  SolverOptions options;
  options.backend = backend;
  options.max_iterations = 5;
  for (auto _ : state) {
    Problem problem = MakeProblem(active_region);
    auto result = sdf_slam::Solve(problem, options);
    benchmark::DoNotOptimize(result.final_cost);
  }
}

void BM_LmDenseMap(benchmark::State& state) { RunBackend(state, SolverBackend::kLm, false); }
void BM_LmActiveRegion(benchmark::State& state) { RunBackend(state, SolverBackend::kLm, true); }
void BM_CeresDenseMap(benchmark::State& state) { RunBackend(state, SolverBackend::kCeres, false); }
void BM_CeresActiveRegion(benchmark::State& state) {
  RunBackend(state, SolverBackend::kCeres, true);
}

}  // namespace

BENCHMARK(BM_LmDenseMap)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_LmActiveRegion)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CeresDenseMap)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CeresActiveRegion)->Unit(benchmark::kMillisecond);

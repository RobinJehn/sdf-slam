#pragma once

#include <filesystem>
#include <vector>

#include "pipeline/config.hpp"
#include "problem/problem.hpp"

namespace sdf_slam {

/// Parses a relations CSV (header line, then i,j,dx,dy,dtheta[,...]) into
/// relation measurements. Trailing residual and inliers columns are ignored;
/// an optional eighth column carries a per-relation weight (squared-weight
/// convention) that loads into sqrt_weight.
std::vector<RelationMeasurement> LoadRelations(const std::filesystem::path& path);

/// Executes a full run (batch or incremental), writes outputs to
/// config.output_dir, and returns the aggregate solver statistics.
SolveResult Run(const RunConfig& config);

}  // namespace sdf_slam

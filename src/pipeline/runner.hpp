#pragma once

#include "pipeline/config.hpp"

namespace sdf_slam {

/// Executes a full run (batch or incremental), writes outputs to
/// config.output_dir, and returns the aggregate solver statistics.
SolveResult Run(const RunConfig& config);

}  // namespace sdf_slam

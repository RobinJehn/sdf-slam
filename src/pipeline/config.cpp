#include "pipeline/config.hpp"

#include <yaml-cpp/yaml.h>

#include <set>
#include <stdexcept>
#include <string>

namespace sdf_slam {

namespace {

void CheckKnownKeys(const YAML::Node& node, const std::set<std::string>& known,
                    const std::string& scope) {
  for (const auto& entry : node) {
    const auto key = entry.first.as<std::string>();
    if (!known.contains(key)) {
      std::string message = "unknown config key '";
      message += scope;
      message += ".";
      message += key;
      message += "'";
      throw std::runtime_error(message);
    }
  }
}

template <typename T>
void Assign(const YAML::Node& node, const char* key, T& target) {
  if (node[key]) {
    target = node[key].as<T>();
  }
}

NormalMethod ParseNormalMethod(const std::string& name) {
  if (name == "pca") {
    return NormalMethod::kPca;
  }
  if (name == "hybrid") {
    return NormalMethod::kHybrid;
  }
  if (name == "weighted") {
    return NormalMethod::kWeighted;
  }
  throw std::runtime_error("unknown normals.method '" + name + "'");
}

SolverBackend ParseBackend(const std::string& name) {
  if (name == "lm") {
    return SolverBackend::kLm;
  }
  if (name == "ceres") {
    return SolverBackend::kCeres;
  }
  throw std::runtime_error("unknown solver.backend '" + name + "'");
}

LinearSolver ParseLinearSolver(const std::string& name) {
  if (name == "eigen") {
    return LinearSolver::kEigen;
  }
  if (name == "cholmod") {
    return LinearSolver::kCholmod;
  }
  throw std::runtime_error("unknown solver.linear_solver '" + name + "'");
}

RunMode ParseMode(const std::string& name) {
  if (name == "batch") {
    return RunMode::kBatch;
  }
  if (name == "incremental") {
    return RunMode::kIncremental;
  }
  throw std::runtime_error("unknown mode '" + name + "'");
}

}  // namespace

RunConfig LoadConfig(const std::filesystem::path& path) {
  const YAML::Node root = YAML::LoadFile(path.string());
  CheckKnownKeys(
      root,
      {"dataset_dir", "ground_truth_poses", "initial_poses", "relations_file", "output_dir", "mode",
       "increment_size", "iterations_per_increment", "snapshot_every", "map", "weights",
       "hallucination", "normals", "huber_delta", "smooth_gradient", "smooth_gradient_step",
       "active_region", "active_margin", "solver"},
      "root");

  RunConfig config;
  if (root["dataset_dir"]) {
    config.dataset_dir = root["dataset_dir"].as<std::string>();
  } else {
    throw std::runtime_error("config requires dataset_dir");
  }
  if (root["ground_truth_poses"]) {
    config.ground_truth_poses = root["ground_truth_poses"].as<std::string>();
  }
  if (root["initial_poses"]) {
    config.initial_poses = root["initial_poses"].as<std::string>();
  }
  if (root["relations_file"]) {
    config.relations_file = root["relations_file"].as<std::string>();
  }
  if (root["output_dir"]) {
    config.output_dir = root["output_dir"].as<std::string>();
  }
  if (root["mode"]) {
    config.mode = ParseMode(root["mode"].as<std::string>());
  }
  Assign(root, "increment_size", config.increment_size);
  Assign(root, "iterations_per_increment", config.iterations_per_increment);
  Assign(root, "snapshot_every", config.snapshot_every);
  Assign(root, "huber_delta", config.problem.huber_delta);
  Assign(root, "smooth_gradient", config.problem.smooth_gradient);
  Assign(root, "smooth_gradient_step", config.problem.smooth_gradient_step);
  Assign(root, "active_region", config.problem.active_region);
  Assign(root, "active_margin", config.problem.active_margin);

  if (const YAML::Node map = root["map"]) {
    CheckKnownKeys(map,
                   {"nx", "ny", "min", "max", "auto_domain", "auto_domain_margin", "initial_value",
                    "init_from_scans", "init_signed"},
                   "map");
    Assign(map, "nx", config.map.nx);
    Assign(map, "ny", config.map.ny);
    if (map["min"]) {
      const auto values = map["min"].as<std::vector<double>>();
      config.map.min_corner = {values.at(0), values.at(1)};
    }
    if (map["max"]) {
      const auto values = map["max"].as<std::vector<double>>();
      config.map.max_corner = {values.at(0), values.at(1)};
    }
    Assign(map, "auto_domain", config.map.auto_domain);
    Assign(map, "auto_domain_margin", config.map.auto_domain_margin);
    Assign(map, "initial_value", config.map.initial_value);
    Assign(map, "init_from_scans", config.map.init_from_scans);
    Assign(map, "init_signed", config.map.init_signed);
  }

  if (const YAML::Node weights = root["weights"]) {
    CheckKnownKeys(weights, {"scan", "hallucination", "eikonal", "odometry", "relation"},
                   "weights");
    Assign(weights, "scan", config.problem.weights.scan);
    Assign(weights, "hallucination", config.problem.weights.hallucination);
    Assign(weights, "eikonal", config.problem.weights.eikonal);
    Assign(weights, "odometry", config.problem.weights.odometry);
    Assign(weights, "relation", config.problem.weights.relation);
  }

  if (const YAML::Node hall = root["hallucination"]) {
    CheckKnownKeys(hall, {"points_per_scan_point", "step_size", "both_directions"},
                   "hallucination");
    Assign(hall, "points_per_scan_point", config.problem.hallucination.points_per_scan_point);
    Assign(hall, "step_size", config.problem.hallucination.step_size);
    Assign(hall, "both_directions", config.problem.hallucination.both_directions);
  }

  if (const YAML::Node normals = root["normals"]) {
    CheckKnownKeys(normals, {"method", "k_neighbors", "corner_threshold", "per_scan"}, "normals");
    if (normals["method"]) {
      config.problem.normals.method = ParseNormalMethod(normals["method"].as<std::string>());
    }
    Assign(normals, "k_neighbors", config.problem.normals.k_neighbors);
    Assign(normals, "corner_threshold", config.problem.normals.corner_threshold);
    Assign(normals, "per_scan", config.problem.normals.per_scan);
  }

  if (const YAML::Node solver = root["solver"]) {
    CheckKnownKeys(
        solver,
        {"backend", "max_iterations", "step_tolerance", "lambda_init", "lambda_factor",
         "reject_worse_steps", "marquardt_scaling", "trust_region", "num_threads", "linear_solver"},
        "solver");
    if (solver["backend"]) {
      config.solver.backend = ParseBackend(solver["backend"].as<std::string>());
    }
    if (solver["linear_solver"]) {
      config.solver.linear_solver = ParseLinearSolver(solver["linear_solver"].as<std::string>());
    }
    Assign(solver, "max_iterations", config.solver.max_iterations);
    Assign(solver, "step_tolerance", config.solver.step_tolerance);
    Assign(solver, "lambda_init", config.solver.lambda_init);
    Assign(solver, "lambda_factor", config.solver.lambda_factor);
    Assign(solver, "reject_worse_steps", config.solver.reject_worse_steps);
    Assign(solver, "marquardt_scaling", config.solver.marquardt_scaling);
    Assign(solver, "trust_region", config.solver.trust_region);
    Assign(solver, "num_threads", config.solver.num_threads);
  }

  return config;
}

}  // namespace sdf_slam

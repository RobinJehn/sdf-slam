#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "pipeline/config.hpp"
#include "pipeline/runner.hpp"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: run_slam <config.yaml> [output_dir]\n";
    return 1;
  }
  try {
    sdf_slam::RunConfig config = sdf_slam::LoadConfig(argv[1]);
    if (argc == 3) {
      config.output_dir = argv[2];
    }
    const sdf_slam::SolveResult result = sdf_slam::Run(config);
    std::cout << "iterations: " << result.iterations << "\n"
              << "initial cost: " << result.initial_cost << "\n"
              << "final cost: " << result.final_cost << "\n"
              << "duration: " << result.duration_seconds << " s\n"
              << "outputs: " << std::filesystem::absolute(config.output_dir).string() << "\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}

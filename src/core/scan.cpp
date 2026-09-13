#include "core/scan.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sdf_slam {

namespace {

/// Returns the whitespace-separated tokens of a line.
std::vector<std::string> Tokenize(const std::string& line) {
  std::istringstream stream(line);
  std::vector<std::string> tokens;
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

}  // namespace

Scan LoadPcd(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open PCD file: " + path.string());
  }

  int x_index = -1;
  int y_index = -1;
  std::string line;
  bool data_section = false;
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const std::vector<std::string> tokens = Tokenize(line);
    if (tokens.empty()) {
      continue;
    }
    if (tokens.front() == "FIELDS") {
      for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "x") {
          x_index = static_cast<int>(i) - 1;
        } else if (tokens[i] == "y") {
          y_index = static_cast<int>(i) - 1;
        }
      }
    } else if (tokens.front() == "DATA") {
      if (tokens.size() < 2 || tokens[1] != "ascii") {
        throw std::runtime_error("only ASCII PCD is supported: " + path.string());
      }
      data_section = true;
      break;
    }
  }
  if (!data_section || x_index < 0 || y_index < 0) {
    throw std::runtime_error("PCD file lacks x/y fields or DATA section: " + path.string());
  }

  Scan scan;
  while (std::getline(file, line)) {
    const std::vector<std::string> tokens = Tokenize(line);
    if (tokens.empty()) {
      continue;
    }
    const auto max_index = static_cast<size_t>(std::max(x_index, y_index));
    if (tokens.size() <= max_index) {
      throw std::runtime_error("malformed PCD data row in " + path.string());
    }
    scan.points.emplace_back(std::stod(tokens[static_cast<size_t>(x_index)]),
                             std::stod(tokens[static_cast<size_t>(y_index)]));
  }
  return scan;
}

std::vector<Pose2> LoadPoses(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open pose file: " + path.string());
  }
  // Accepts both pose formats: whitespace-separated (scanner_info.txt) and
  // comma-separated with a header line (poses_*.csv run outputs).
  std::vector<Pose2> poses;
  std::string line;
  while (std::getline(file, line)) {
    std::ranges::replace(line, ',', ' ');
    std::istringstream stream(line);
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
    if (stream >> x >> y >> theta) {
      poses.push_back({x, y, theta});
    }
  }
  return poses;
}

Dataset LoadDataset(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> scan_files;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() == ".pcd") {
      scan_files.push_back(entry.path());
    }
  }
  // Filename sort keeps scan00 before scan01 (zero-padded names).
  std::ranges::sort(scan_files);

  Dataset dataset;
  dataset.scans.reserve(scan_files.size());
  for (const auto& file : scan_files) {
    dataset.scans.push_back(LoadPcd(file));
  }
  dataset.poses = LoadPoses(dir / "scanner_info.txt");
  if (dataset.poses.size() != dataset.scans.size()) {
    throw std::runtime_error("scan count (" + std::to_string(dataset.scans.size()) +
                             ") != pose count (" + std::to_string(dataset.poses.size()) + ") in " +
                             dir.string());
  }
  return dataset;
}

}  // namespace sdf_slam

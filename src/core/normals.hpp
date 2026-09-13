#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <vector>

#include "core/grid_map.hpp"
#include "core/pose.hpp"
#include "core/scan.hpp"

namespace sdf_slam {

/// Scan points in the global frame with per-point surface normals.
/// Normals are oriented toward the origin of the scan they belong to, so they
/// point away from the surface on the observed side (appendix A.1.1.3).
struct ScanNormals {
  std::vector<Eigen::Vector2d> points;
  std::vector<Eigen::Vector2d> normals;
  /// Eigenvalue ratio lambda_min / lambda_max of the PCA neighborhood in
  /// [0, 1]. Near 0 on a clean straight wall; grows when the neighborhood
  /// spans a corner or several surfaces.
  std::vector<double> cornerness;
};

/// Estimates a normal per scan point via k-nearest-neighbor PCA. With
/// per_scan false the neighborhoods span the union of all scans transformed
/// with the given poses; with per_scan true each scan is processed alone in
/// its sensor frame and the normals are rotated by the pose.
ScanNormals ComputeScanNormals(const std::vector<Scan>& scans, const std::vector<Pose2>& poses,
                               int k_neighbors, bool per_scan = false);

/// How grid-node normals are derived from scan-point normals.
/// see DEC-0003 normal-estimation-ablation
enum class NormalMethod : std::uint8_t {
  /// Nearest scan point's PCA normal (dissertation baseline).
  kPca,
  /// Blend the PCA normal with the sign-matched vector toward the nearest
  /// scan point where the PCA neighborhood looks corner-like.
  kHybrid,
  /// PCA normal, but scale the Eikonal weight down with cornerness.
  kWeighted,
};

struct NormalOptions {
  /// `kWeighted` wins the ablation: 3x lower relative rotation error under
  /// odometry noise than the PCA baseline. see DEC-0003 normal-estimation-ablation
  NormalMethod method{NormalMethod::kWeighted};
  int k_neighbors{10};
  /// When true, PCA neighborhoods use only points of the same scan, computed
  /// in the sensor frame and rotated by the pose. Per-scan normals are
  /// pose-drift-independent; global neighborhoods mix points from misaligned
  /// frames and corrupt the normals exactly when the estimate drifts.
  bool per_scan{false};
  /// Cornerness at which the hybrid blend saturates / the weighted scale
  /// reaches its floor.
  double corner_threshold{0.2};
};

/// Per grid node: the normal used in the Eikonal residual plus a scale factor
/// for that residual's weight (1 except for NormalMethod::kWeighted).
struct NodeNormal {
  Eigen::Vector2d normal{0.0, 0.0};
  double eikonal_scale{1.0};
};

/// Assigns a normal to every grid node from the nearest scan point, applying
/// the configured method.
std::vector<NodeNormal> AssignNodeNormals(const GridMap& map, const ScanNormals& scan_normals,
                                          const NormalOptions& options);

}  // namespace sdf_slam

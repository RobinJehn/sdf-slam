#include "core/normals.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <nanoflann.hpp>
#include <stdexcept>

namespace sdf_slam {

namespace {

/// nanoflann adaptor over a flat vector of 2D points.
struct PointCloudAdaptor {
  const std::vector<Eigen::Vector2d>* points;

  [[nodiscard]] size_t kdtree_get_point_count() const { return points->size(); }
  [[nodiscard]] double kdtree_get_pt(size_t idx, size_t dim) const {
    return (*points)[idx][static_cast<Eigen::Index>(dim)];
  }
  template <class Bbox>
  bool kdtree_get_bbox(Bbox& /*bbox*/) const {
    return false;
  }
};

using KdTree =
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
                                        PointCloudAdaptor, 2>;

}  // namespace

ScanNormals ComputeScanNormals(const std::vector<Scan>& scans, const std::vector<Pose2>& poses,
                               int k_neighbors) {
  if (scans.size() != poses.size()) {
    throw std::invalid_argument("scan count != pose count");
  }

  ScanNormals result;
  std::vector<Eigen::Vector2d> origins;  // scan origin per global point
  for (size_t i = 0; i < scans.size(); ++i) {
    for (const auto& p : scans[i].points) {
      result.points.push_back(poses[i].Apply(p));
      origins.push_back(poses[i].Translation());
    }
  }
  const size_t n = result.points.size();
  result.normals.resize(n);
  result.cornerness.resize(n);
  if (n == 0) {
    return result;
  }

  const PointCloudAdaptor adaptor{&result.points};
  const KdTree tree(2, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));

  const size_t k = static_cast<size_t>(std::max(2, k_neighbors));
  std::vector<uint32_t> indices(k);
  std::vector<double> distances(k);
  for (size_t i = 0; i < n; ++i) {
    const size_t found =
        tree.knnSearch(result.points[i].data(), k, indices.data(), distances.data());

    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (size_t j = 0; j < found; ++j) {
      mean += result.points[indices[j]];
    }
    mean /= static_cast<double>(found);

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (size_t j = 0; j < found; ++j) {
      const Eigen::Vector2d d = result.points[indices[j]] - mean;
      cov += d * d.transpose();
    }

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
    // Eigenvalues sorted ascending: [0] is the normal direction.
    Eigen::Vector2d normal = solver.eigenvectors().col(0);
    const double lambda_min = std::max(solver.eigenvalues()[0], 0.0);
    const double lambda_max = std::max(solver.eigenvalues()[1], 1e-12);

    // Orient toward the scan origin so the normal points into observed free
    // space (appendix A.1.1.3).
    if (normal.dot(origins[i] - result.points[i]) < 0.0) {
      normal = -normal;
    }
    result.normals[i] = normal;
    result.cornerness[i] = lambda_min / lambda_max;
  }
  return result;
}

std::vector<NodeNormal> AssignNodeNormals(const GridMap& map, const ScanNormals& scan_normals,
                                          const NormalOptions& options) {
  std::vector<NodeNormal> node_normals(static_cast<size_t>(map.num_nodes()));
  if (scan_normals.points.empty()) {
    return node_normals;
  }

  const PointCloudAdaptor adaptor{&scan_normals.points};
  const KdTree tree(2, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));

  for (int h = 0; h < map.ny(); ++h) {
    for (int w = 0; w < map.nx(); ++w) {
      const Eigen::Vector2d node = map.NodePosition(w, h);
      uint32_t nearest = 0;
      double distance_sq = 0.0;
      tree.knnSearch(node.data(), 1, &nearest, &distance_sq);

      const Eigen::Vector2d n_pca = scan_normals.normals[nearest];
      const double corner =
          std::clamp(scan_normals.cornerness[nearest] / options.corner_threshold, 0.0, 1.0);

      NodeNormal out;
      switch (options.method) {
        case NormalMethod::kPca:
          out.normal = n_pca;
          break;
        case NormalMethod::kHybrid: {
          // Closest-point cue: direction of the segment node -> surface,
          // sign-matched to the PCA normal. Near corners the PCA neighborhood
          // spans two walls and its normal tilts; the closest-point vector
          // still points at the actual nearest surface (section 6.3).
          const Eigen::Vector2d to_surface = scan_normals.points[nearest] - node;
          const double norm = to_surface.norm();
          if (norm < 1e-9) {
            out.normal = n_pca;
            break;
          }
          Eigen::Vector2d cue = to_surface / norm;
          if (cue.dot(n_pca) < 0.0) {
            cue = -cue;
          }
          const Eigen::Vector2d blended = (1.0 - corner) * n_pca + corner * cue;
          const double blended_norm = blended.norm();
          out.normal = blended_norm > 1e-9 ? Eigen::Vector2d(blended / blended_norm) : n_pca;
          break;
        }
        case NormalMethod::kWeighted:
          out.normal = n_pca;
          // Unreliable normals should not force a wrong gradient; the Eikonal
          // residual fades out as the neighborhood turns corner-like.
          out.eikonal_scale = 1.0 - corner;
          break;
      }
      node_normals[static_cast<size_t>(map.NodeId(w, h))] = out;
    }
  }
  return node_normals;
}

}  // namespace sdf_slam

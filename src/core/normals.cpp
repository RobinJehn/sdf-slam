#include "core/normals.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <functional>
#include <nanoflann.hpp>
#include <stdexcept>
#include <thread>

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

/// Runs fn(i) for i in [0, n) on hardware-concurrency threads. Each index
/// writes only its own output slot, so the result does not depend on the
/// thread count or schedule.
void ParallelFor(size_t n, const std::function<void(size_t)>& fn) {
  const size_t num_threads = std::min<size_t>(n, std::max(1U, std::thread::hardware_concurrency()));
  if (num_threads <= 1) {
    for (size_t i = 0; i < n; ++i) {
      fn(i);
    }
    return;
  }
  const size_t chunk = (n + num_threads - 1) / num_threads;
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (size_t t = 0; t < num_threads; ++t) {
    const size_t begin = t * chunk;
    const size_t end = std::min(n, begin + chunk);
    if (begin >= end) {
      break;
    }
    workers.emplace_back([begin, end, &fn] {
      for (size_t i = begin; i < end; ++i) {
        fn(i);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

}  // namespace

namespace {

/// PCA normal + cornerness from the k nearest neighbors of point i inside
/// `points`; the normal is oriented toward `origin`.
void PcaNormal(const std::vector<Eigen::Vector2d>& points, const KdTree& tree, size_t i, size_t k,
               const Eigen::Vector2d& origin, Eigen::Vector2d& normal_out, double& cornerness_out) {
  std::vector<uint32_t> indices(k);
  std::vector<double> distances(k);
  const size_t found = tree.knnSearch(points[i].data(), k, indices.data(), distances.data());

  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  for (size_t j = 0; j < found; ++j) {
    mean += points[indices[j]];
  }
  mean /= static_cast<double>(found);

  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (size_t j = 0; j < found; ++j) {
    const Eigen::Vector2d d = points[indices[j]] - mean;
    cov += d * d.transpose();
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
  // Eigenvalues sorted ascending: [0] is the normal direction.
  Eigen::Vector2d normal = solver.eigenvectors().col(0);
  const double lambda_min = std::max(solver.eigenvalues()[0], 0.0);
  const double lambda_max = std::max(solver.eigenvalues()[1], 1e-12);

  // Orient toward the scan origin so the normal points into observed free
  // space (appendix A.1.1.3).
  if (normal.dot(origin - points[i]) < 0.0) {
    normal = -normal;
  }
  normal_out = normal;
  cornerness_out = lambda_min / lambda_max;
}

ScanNormals ComputePerScanNormals(const std::vector<Scan>& scans, const std::vector<Pose2>& poses,
                                  size_t k) {
  ScanNormals result;
  for (size_t s = 0; s < scans.size(); ++s) {
    const std::vector<Eigen::Vector2d>& local = scans[s].points;
    const PointCloudAdaptor adaptor{&local};
    const KdTree tree(2, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    const Eigen::Matrix2d rot = poses[s].Rotation();
    for (size_t i = 0; i < local.size(); ++i) {
      Eigen::Vector2d normal;
      double cornerness = 0.0;
      // Sensor origin is (0, 0) in the scan frame.
      PcaNormal(local, tree, i, std::min(k, local.size()), Eigen::Vector2d::Zero(), normal,
                cornerness);
      result.points.push_back(poses[s].Apply(local[i]));
      result.normals.emplace_back(rot * normal);
      result.cornerness.push_back(cornerness);
    }
  }
  return result;
}

}  // namespace

ScanNormals ComputeScanNormals(const std::vector<Scan>& scans, const std::vector<Pose2>& poses,
                               int k_neighbors, bool per_scan) {
  if (scans.size() != poses.size()) {
    throw std::invalid_argument("scan count != pose count");
  }
  const size_t k = static_cast<size_t>(std::max(2, k_neighbors));
  if (per_scan) {
    return ComputePerScanNormals(scans, poses, k);
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

  ParallelFor(n, [&](size_t i) {
    PcaNormal(result.points, tree, i, std::min(k, n), origins[i], result.normals[i],
              result.cornerness[i]);
  });
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

  ParallelFor(static_cast<size_t>(map.ny()), [&](size_t row) {
    const int h = static_cast<int>(row);
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
  });
  return node_normals;
}

}  // namespace sdf_slam

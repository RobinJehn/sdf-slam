#pragma once

#include <Eigen/Core>
#include <array>
#include <cassert>

namespace sdf_slam {

/// A discrete SDF grid over a rectangular domain with bilinear interpolation
/// between nodes. Node (w, h) has flat id h * nx + w.
class GridMap {
 public:
  GridMap(int nx, int ny, const Eigen::Vector2d& min_corner, const Eigen::Vector2d& max_corner,
          double initial_value = 0.0)
      : nx_(nx),
        ny_(ny),
        min_(min_corner),
        dx_((max_corner.x() - min_corner.x()) / (nx - 1)),
        dy_((max_corner.y() - min_corner.y()) / (ny - 1)),
        values_(Eigen::VectorXd::Constant(static_cast<Eigen::Index>(nx) * ny, initial_value)) {
    assert(nx >= 2 && ny >= 2);
    assert(dx_ > 0.0 && dy_ > 0.0);
  }

  [[nodiscard]] int nx() const { return nx_; }
  [[nodiscard]] int ny() const { return ny_; }
  [[nodiscard]] double dx() const { return dx_; }
  [[nodiscard]] double dy() const { return dy_; }
  [[nodiscard]] Eigen::Vector2d min_corner() const { return min_; }
  [[nodiscard]] int num_nodes() const { return nx_ * ny_; }

  [[nodiscard]] int NodeId(int w, int h) const { return h * nx_ + w; }

  [[nodiscard]] Eigen::Vector2d NodePosition(int w, int h) const {
    return {min_.x() + w * dx_, min_.y() + h * dy_};
  }

  [[nodiscard]] double Value(int w, int h) const { return values_[NodeId(w, h)]; }
  double& Value(int w, int h) { return values_[NodeId(w, h)]; }

  [[nodiscard]] const Eigen::VectorXd& values() const { return values_; }
  Eigen::VectorXd& values() { return values_; }

  /// The grid cell containing a query point, with interpolation weights.
  /// Node order in `node_ids`/`weights`: (w,h), (w+1,h), (w,h+1), (w+1,h+1).
  struct CellRef {
    int w{0};
    int h{0};
    double alpha{0.0};
    double beta{0.0};
    std::array<int, 4> node_ids{};
    std::array<double, 4> weights{};
  };

  /// Locates the cell containing p. Points on a grid line belong to the cell
  /// to the right/top; the last row/column falls back to the cell below/left.
  /// Returns false when p lies outside the domain.
  [[nodiscard]] bool Locate(const Eigen::Vector2d& p, CellRef& cell) const {
    const double fx = (p.x() - min_.x()) / dx_;
    const double fy = (p.y() - min_.y()) / dy_;
    if (fx < 0.0 || fy < 0.0 || fx > nx_ - 1 || fy > ny_ - 1) {
      return false;
    }
    cell.w = std::min(static_cast<int>(fx), nx_ - 2);
    cell.h = std::min(static_cast<int>(fy), ny_ - 2);
    cell.alpha = fx - cell.w;
    cell.beta = fy - cell.h;
    FillCell(cell);
    return true;
  }

  /// Like Locate, but clamps alpha/beta into the boundary cell so a point
  /// slightly outside the domain extrapolates the nearest cell's bilinear
  /// patch instead of being dropped.
  void LocateClamped(const Eigen::Vector2d& p, CellRef& cell) const {
    const double fx = (p.x() - min_.x()) / dx_;
    const double fy = (p.y() - min_.y()) / dy_;
    cell.w = std::clamp(static_cast<int>(fx), 0, nx_ - 2);
    cell.h = std::clamp(static_cast<int>(fy), 0, ny_ - 2);
    cell.alpha = fx - cell.w;
    cell.beta = fy - cell.h;
    FillCell(cell);
  }

  /// Bilinear interpolation (dissertation eq. 3.7).
  [[nodiscard]] double Interpolate(const CellRef& cell) const {
    double value = 0.0;
    for (int i = 0; i < 4; ++i) {
      value +=
          cell.weights[static_cast<size_t>(i)] * values_[cell.node_ids[static_cast<size_t>(i)]];
    }
    return value;
  }

  /// Analytic spatial gradient of the bilinear patch at the cell's
  /// (alpha, beta). Exact derivative of Interpolate, so residual Jacobians
  /// stay consistent with the interpolation model.
  [[nodiscard]] Eigen::Vector2d Gradient(const CellRef& cell) const {
    const double d00 = values_[cell.node_ids[0]];
    const double d10 = values_[cell.node_ids[1]];
    const double d01 = values_[cell.node_ids[2]];
    const double d11 = values_[cell.node_ids[3]];
    const double gx = ((1.0 - cell.beta) * (d10 - d00) + cell.beta * (d11 - d01)) / dx_;
    const double gy = ((1.0 - cell.alpha) * (d01 - d00) + cell.alpha * (d11 - d10)) / dy_;
    return {gx, gy};
  }

 private:
  void FillCell(CellRef& cell) const {
    cell.node_ids = {NodeId(cell.w, cell.h), NodeId(cell.w + 1, cell.h), NodeId(cell.w, cell.h + 1),
                     NodeId(cell.w + 1, cell.h + 1)};
    const double a = cell.alpha;
    const double b = cell.beta;
    cell.weights = {(1.0 - a) * (1.0 - b), a * (1.0 - b), (1.0 - a) * b, a * b};
  }

  int nx_;
  int ny_;
  Eigen::Vector2d min_;
  double dx_;
  double dy_;
  Eigen::VectorXd values_;
};

}  // namespace sdf_slam

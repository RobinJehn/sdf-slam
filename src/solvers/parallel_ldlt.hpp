#pragma once

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <queue>
#include <thread>
#include <vector>

namespace sdf_slam {

/// Simplicial LDLT with a parallel numeric factorization.
///
/// The numeric kernel replicates Eigen's
/// SimplicialCholeskyBase::factorize_preordered (MPL-2.0, itself adapted from
/// Timothy A. Davis' LDL) operation for operation. Rows are scheduled over
/// independent elimination-tree subtrees:
///  - Row k only reads and writes columns that are descendants of k in the
///    elimination tree, so complete disjoint subtrees never touch the same
///    data and run concurrently without synchronization.
///  - The nodes cut off while balancing the subtrees (the "trunk", ancestors
///    of the chosen subtree roots) run serially after all subtrees join.
///  - Appends into any column i of L come from the ancestor chain of i,
///    which is totally ordered, so the append order equals the serial order.
/// Per-row arithmetic and scheduling order are therefore identical to the
/// serial algorithm: the factor is bitwise equal to Eigen's SimplicialLDLT,
/// independent of the thread count.
/// see DEC-0008 bitwise-exact-performance-work
class ParallelSimplicialLdlt : public Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> {
 public:
  using Base = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
  using SpMat = Eigen::SparseMatrix<double>;

  /// Symbolic analysis plus the subtree schedule for `num_threads` workers.
  void AnalyzePatternWithSchedule(const SpMat& a, std::size_t num_threads) {
    Base::analyzePattern(a);
    BuildSchedule(num_threads);
  }

  /// Numeric factorization. Requires AnalyzePatternWithSchedule first.
  void FactorizeParallel(const SpMat& a) {
    eigen_assert(m_analysisIsOk && "AnalyzePatternWithSchedule must run first");
    // Permute the input exactly like Eigen::SimplicialCholeskyBase::factorize.
    SpMat tmp(a.rows(), a.cols());
    Eigen::internal::permute_symm_to_symm<Eigen::Lower, Eigen::Upper, false>(a, tmp,
                                                                             m_P.indices().data());
    FactorizeNumeric(tmp);
  }

 private:
  // Rows each worker processes in phase 1 (ascending within each list), and
  // the serial phase-2 rows (ascending).
  std::vector<std::vector<int>> thread_rows_;
  std::vector<int> trunk_rows_;

  void BuildSchedule(std::size_t num_threads) {
    const int n = static_cast<int>(m_parent.size());
    thread_rows_.assign(std::max<std::size_t>(1, num_threads), {});
    trunk_rows_.clear();
    if (num_threads <= 1 || n < 256) {
      // Single worker: everything in one ascending list, no trunk.
      auto& rows = thread_rows_[0];
      rows.resize(static_cast<std::size_t>(n));
      for (int i = 0; i < n; ++i) {
        rows[static_cast<std::size_t>(i)] = i;
      }
      thread_rows_.resize(1);
      return;
    }

    // Children lists and subtree sizes. The elimination tree satisfies
    // parent(i) > i, so one ascending pass accumulates sizes bottom-up.
    std::vector<std::vector<int>> children(static_cast<std::size_t>(n));
    std::vector<int> subtree_size(static_cast<std::size_t>(n), 1);
    std::vector<int> roots;
    for (int i = 0; i < n; ++i) {
      const int p = m_parent[i];
      if (p >= 0) {
        children[static_cast<std::size_t>(p)].push_back(i);
        // Children of p all have smaller indices, so sizes below i are final.
      } else {
        roots.push_back(i);
      }
    }
    for (int i = 0; i < n; ++i) {
      const int p = m_parent[i];
      if (p >= 0) {
        subtree_size[static_cast<std::size_t>(p)] += subtree_size[static_cast<std::size_t>(i)];
      }
    }

    // Split the largest candidate subtree until the pieces balance across
    // the workers. Split roots move to the serial trunk.
    const auto by_size = [&subtree_size](int a, int b) {
      return subtree_size[static_cast<std::size_t>(a)] < subtree_size[static_cast<std::size_t>(b)];
    };
    std::priority_queue<int, std::vector<int>, decltype(by_size)> heap(by_size);
    for (const int r : roots) {
      heap.push(r);
    }
    std::vector<int> subtree_roots;
    const int target = std::max(1, n / static_cast<int>(num_threads * 8));
    while (!heap.empty()) {
      const int top = heap.top();
      const bool splittable = !children[static_cast<std::size_t>(top)].empty();
      if (subtree_size[static_cast<std::size_t>(top)] <= target ||
          subtree_roots.size() + heap.size() >= num_threads * 16 || !splittable) {
        break;
      }
      heap.pop();
      trunk_rows_.push_back(top);
      for (const int c : children[static_cast<std::size_t>(top)]) {
        heap.push(c);
      }
    }
    while (!heap.empty()) {
      subtree_roots.push_back(heap.top());
      heap.pop();
    }

    // Longest-processing-time assignment of subtrees to workers.
    std::ranges::sort(subtree_roots, [&](int a, int b) {
      return subtree_size[static_cast<std::size_t>(a)] > subtree_size[static_cast<std::size_t>(b)];
    });
    std::vector<long> load(num_threads, 0);
    std::vector<int> stack;
    for (const int r : subtree_roots) {
      const std::size_t t = static_cast<std::size_t>(std::ranges::min_element(load) - load.begin());
      load[t] += subtree_size[static_cast<std::size_t>(r)];
      // Collect the subtree nodes.
      stack.push_back(r);
      while (!stack.empty()) {
        const int v = stack.back();
        stack.pop_back();
        thread_rows_[t].push_back(v);
        for (const int c : children[static_cast<std::size_t>(v)]) {
          stack.push_back(c);
        }
      }
    }
    for (auto& rows : thread_rows_) {
      std::ranges::sort(rows);
    }
    std::ranges::sort(trunk_rows_);
  }

  void FactorizeNumeric(const SpMat& ap) {
    const int size = static_cast<int>(ap.rows());
    const int* lp = m_matrix.outerIndexPtr();
    int* li = m_matrix.innerIndexPtr();
    double* lx = m_matrix.valuePtr();
    m_diag.resize(size);
    int* nonzeros_per_col = m_workSpace.data();

    std::atomic<bool> ok{true};

    // One row of L, exactly Eigen's factorize_preordered loop body (LDLT
    // branch, real scalars). Floating-point contraction is pinned to the
    // instruction selection of the reference build: the shift term and the
    // y-updates are fused (std::fma), the diagonal update is not. The pragma
    // stops the compiler from re-fusing the deliberately unfused product.
    const auto process_row = [&](int k, double* y, int* pattern, int* tags) {
#pragma STDC FP_CONTRACT OFF
      y[k] = 0.0;
      int top = size;
      tags[k] = k;
      nonzeros_per_col[k] = 0;
      for (SpMat::InnerIterator it(ap, k); it; ++it) {
        int i = static_cast<int>(it.index());
        if (i <= k) {
          y[i] += it.value();
          int len = 0;
          for (; tags[i] != k; i = m_parent[i]) {
            pattern[len++] = i;
            tags[i] = k;
          }
          while (len > 0) {
            pattern[--top] = pattern[--len];
          }
        }
      }

      double d = std::fma(y[k], m_shiftScale, m_shiftOffset);
      y[k] = 0.0;
      for (; top < size; ++top) {
        const int i = pattern[top];
        const double yi = y[i];
        y[i] = 0.0;
        const double l_ki = yi / m_diag[i];
        const int p2 = lp[i] + nonzeros_per_col[i];
        int p = lp[i];
        for (; p < p2; ++p) {
          y[li[p]] = std::fma(-lx[p], yi, y[li[p]]);
        }
        const double diag_update = l_ki * yi;
        d -= diag_update;
        li[p] = k;
        lx[p] = l_ki;
        ++nonzeros_per_col[i];
      }
      m_diag[k] = d;
      return d != 0.0;
    };

    const auto process_rows = [&](const std::vector<int>& rows) {
      std::vector<double> y(static_cast<std::size_t>(size), 0.0);
      std::vector<int> pattern(static_cast<std::size_t>(size), 0);
      std::vector<int> tags(static_cast<std::size_t>(size), -1);
      for (const int k : rows) {
        if (!ok.load(std::memory_order_relaxed)) {
          return;
        }
        if (!process_row(k, y.data(), pattern.data(), tags.data())) {
          ok.store(false, std::memory_order_relaxed);
          return;
        }
      }
    };

    if (thread_rows_.size() <= 1) {
      if (!thread_rows_.empty()) {
        process_rows(thread_rows_[0]);
      }
    } else {
      std::vector<std::thread> workers;
      workers.reserve(thread_rows_.size());
      for (const auto& rows : thread_rows_) {
        workers.emplace_back(process_rows, std::cref(rows));
      }
      for (auto& worker : workers) {
        worker.join();
      }
    }
    if (ok.load()) {
      process_rows(trunk_rows_);
    }

    // m_isInitialized is already true after analyzePattern.
    m_info = ok.load() ? Eigen::Success : Eigen::NumericalIssue;
    m_factorizationIsOk = true;
  }
};

}  // namespace sdf_slam

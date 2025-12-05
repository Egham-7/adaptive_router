#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <limits>

#include "cluster_backend.hpp"

namespace adaptive {

// CPU backend using Eigen for cluster assignment
class CpuClusterBackend : public IClusterBackend {
public:
  CpuClusterBackend() = default;
  ~CpuClusterBackend() override = default;

  void load_centroids(const float* data, int n_clusters, int dim) override {
    n_clusters_ = n_clusters;
    dim_ = dim;

    // Map raw data to Eigen matrix (row-major input, store as Eigen default)
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input(
        data, n_clusters, dim);
    centroids_ = input;
  }

  [[nodiscard]] std::pair<int, float> assign(const float* embedding, int dim) override {
    if (n_clusters_ == 0 || dim != dim_) {
      return {-1, 0.0f};
    }

    // Map embedding to Eigen vector (zero-copy)
    Eigen::Map<const Eigen::VectorXf> emb(embedding, dim);

    // Find nearest centroid using vectorized operations
    int best_cluster = 0;
    float best_distance = std::numeric_limits<float>::max();

    for (int i = 0; i < n_clusters_; ++i) {
      float dist = (centroids_.row(i).transpose() - emb).squaredNorm();
      if (dist < best_distance) {
        best_distance = dist;
        best_cluster = i;
      }
    }

    return {best_cluster, std::sqrt(best_distance)};
  }

  [[nodiscard]] int get_n_clusters() const noexcept override { return n_clusters_; }

  [[nodiscard]] int get_dim() const noexcept override { return dim_; }

  [[nodiscard]] bool is_gpu_accelerated() const noexcept override { return false; }

private:
  Eigen::MatrixXf centroids_;
  int n_clusters_ = 0;
  int dim_ = 0;
};

}  // namespace adaptive

#pragma once
#include <memory>
#include <utility>

#include "cluster_backend.hpp"
#include "types.hpp"

class ClusterEngine {
public:
  // Default constructor (auto-detect GPU)
  ClusterEngine();

  // Explicit backend selection
  explicit ClusterEngine(adaptive::ClusterBackendType backend_type);

  ~ClusterEngine() = default;

  // Movable
  ClusterEngine(ClusterEngine&&) noexcept = default;
  ClusterEngine& operator=(ClusterEngine&&) noexcept = default;
  ClusterEngine(const ClusterEngine&) = delete;
  ClusterEngine& operator=(const ClusterEngine&) = delete;

  // Load K-means cluster centers (K x D matrix)
  void load_centroids(const EmbeddingMatrix& centers);

  // Assign embedding to nearest cluster
  // Returns (cluster_id, distance) pair
  [[nodiscard]] std::pair<int, float> assign(const EmbeddingVector& embedding);

  // Get number of clusters
  [[nodiscard]] int get_n_clusters() const noexcept;

  // Check if using GPU acceleration
  [[nodiscard]] bool is_gpu_accelerated() const noexcept;

private:
  std::unique_ptr<adaptive::IClusterBackend> backend_;
  int dim_ = 0;
};

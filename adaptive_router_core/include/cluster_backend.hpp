#pragma once
#include <memory>
#include <utility>

namespace adaptive {

// Backend type enumeration
enum class ClusterBackendType { CPU, CUDA, Auto };

// Abstract interface for cluster assignment backends
class IClusterBackend {
public:
  virtual ~IClusterBackend() = default;

  // Non-copyable, non-movable (polymorphic base class)
  IClusterBackend(const IClusterBackend&) = delete;
  IClusterBackend& operator=(const IClusterBackend&) = delete;
  IClusterBackend(IClusterBackend&&) = delete;
  IClusterBackend& operator=(IClusterBackend&&) = delete;

  // Load cluster centroids (n_clusters x dim matrix in row-major order)
  virtual void load_centroids(const float* data, int n_clusters, int dim) = 0;

  // Assign embedding to nearest cluster
  // Returns (cluster_id, distance) pair
  [[nodiscard]] virtual std::pair<int, float> assign(const float* embedding, int dim) = 0;

  // Get number of clusters
  [[nodiscard]] virtual int get_n_clusters() const noexcept = 0;

  // Get embedding dimension
  [[nodiscard]] virtual int get_dim() const noexcept = 0;

  // Check if backend is GPU-accelerated
  [[nodiscard]] virtual bool is_gpu_accelerated() const noexcept = 0;

protected:
  IClusterBackend() = default;
};

// Factory function to create appropriate backend
[[nodiscard]] std::unique_ptr<IClusterBackend> create_cluster_backend(
    ClusterBackendType type = ClusterBackendType::Auto);

// Check if CUDA is available at runtime
[[nodiscard]] bool cuda_available() noexcept;

}  // namespace adaptive

#pragma once

#ifdef ADAPTIVE_HAS_CUDA

#include <cuda_runtime.h>
#include <utility>

#include "cluster_backend.hpp"

namespace adaptive {

// CUDA backend for GPU-accelerated cluster assignment
class CudaClusterBackend : public IClusterBackend {
public:
  CudaClusterBackend();
  ~CudaClusterBackend() override;

  // Non-copyable, non-movable
  CudaClusterBackend(const CudaClusterBackend&) = delete;
  CudaClusterBackend& operator=(const CudaClusterBackend&) = delete;
  CudaClusterBackend(CudaClusterBackend&&) = delete;
  CudaClusterBackend& operator=(CudaClusterBackend&&) = delete;

  void load_centroids(const float* data, int n_clusters, int dim) override;

  [[nodiscard]] std::pair<int, float> assign(const float* embedding, int dim) override;

  [[nodiscard]] int get_n_clusters() const noexcept override { return n_clusters_; }

  [[nodiscard]] int get_dim() const noexcept override { return dim_; }

  [[nodiscard]] bool is_gpu_accelerated() const noexcept override { return true; }

private:
  // Device memory pointers
  float* d_centroids_ = nullptr;   // [n_clusters x dim]
  float* d_embedding_ = nullptr;   // [dim]
  float* d_distances_ = nullptr;   // [n_clusters]

  int n_clusters_ = 0;
  int dim_ = 0;

  // CUDA stream for async operations
  cudaStream_t stream_ = nullptr;

  void free_device_memory();
};

}  // namespace adaptive

#endif  // ADAPTIVE_HAS_CUDA

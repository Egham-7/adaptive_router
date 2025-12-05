#ifdef ADAPTIVE_HAS_CUDA

#include "cluster_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace adaptive {

// CUDA error checking macro
#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t err = (call);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));             \
    }                                                                                              \
  } while (0)

// Kernel: Compute squared Euclidean distances from embedding to all centroids
// Each thread handles one centroid
__global__ void compute_distances_kernel(const float* __restrict__ centroids,
                                         const float* __restrict__ embedding, float* distances,
                                         int n_clusters, int dim) {
  int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cluster_idx >= n_clusters) return;

  float dist = 0.0f;
  const float* centroid = centroids + cluster_idx * dim;

  // Compute squared Euclidean distance
  for (int d = 0; d < dim; ++d) {
    float diff = centroid[d] - embedding[d];
    dist += diff * diff;
  }

  distances[cluster_idx] = dist;
}

CudaClusterBackend::CudaClusterBackend() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

CudaClusterBackend::~CudaClusterBackend() {
  free_device_memory();
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

void CudaClusterBackend::free_device_memory() {
  if (d_centroids_) {
    cudaFree(d_centroids_);
    d_centroids_ = nullptr;
  }
  if (d_embedding_) {
    cudaFree(d_embedding_);
    d_embedding_ = nullptr;
  }
  if (d_distances_) {
    cudaFree(d_distances_);
    d_distances_ = nullptr;
  }
}

void CudaClusterBackend::load_centroids(const float* data, int n_clusters, int dim) {
  // Free existing memory
  free_device_memory();

  n_clusters_ = n_clusters;
  dim_ = dim;

  // Allocate device memory for centroids
  size_t centroids_size = static_cast<size_t>(n_clusters) * dim * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_centroids_, centroids_size));

  // Copy centroids to device (one-time operation)
  CUDA_CHECK(cudaMemcpy(d_centroids_, data, centroids_size, cudaMemcpyHostToDevice));

  // Pre-allocate buffers for embedding and distances
  CUDA_CHECK(cudaMalloc(&d_embedding_, static_cast<size_t>(dim) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_distances_, static_cast<size_t>(n_clusters) * sizeof(float)));
}

std::pair<int, float> CudaClusterBackend::assign(const float* embedding, int dim) {
  if (n_clusters_ == 0 || dim != dim_) {
    return {-1, 0.0f};
  }

  // Copy embedding to device
  CUDA_CHECK(cudaMemcpyAsync(d_embedding_, embedding, static_cast<size_t>(dim) * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));

  // Launch kernel to compute distances
  constexpr int kBlockSize = 256;
  int grid_size = (n_clusters_ + kBlockSize - 1) / kBlockSize;

  compute_distances_kernel<<<grid_size, kBlockSize, 0, stream_>>>(d_centroids_, d_embedding_,
                                                                   d_distances_, n_clusters_, dim_);

  // Synchronize before using Thrust
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Use Thrust to find minimum distance and its index
  thrust::device_ptr<float> d_ptr(d_distances_);
  auto min_it = thrust::min_element(d_ptr, d_ptr + n_clusters_);

  int best_cluster = static_cast<int>(min_it - d_ptr);
  float best_distance_sq = *min_it;

  return {best_cluster, std::sqrt(best_distance_sq)};
}

}  // namespace adaptive

#endif  // ADAPTIVE_HAS_CUDA

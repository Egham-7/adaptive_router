#include "cluster_backend.hpp"

#include "cluster_cpu.hpp"

#ifdef ADAPTIVE_HAS_CUDA
#include "cuda/cluster_cuda.hpp"
#include <cuda_runtime.h>
#endif

namespace adaptive {

bool cuda_available() noexcept {
#ifdef ADAPTIVE_HAS_CUDA
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
#else
  return false;
#endif
}

std::unique_ptr<IClusterBackend> create_cluster_backend(ClusterBackendType type) {
  switch (type) {
    case ClusterBackendType::CPU:
      return std::make_unique<CpuClusterBackend>();

    case ClusterBackendType::CUDA:
#ifdef ADAPTIVE_HAS_CUDA
      if (cuda_available()) {
        return std::make_unique<CudaClusterBackend>();
      }
#endif
      // Fall back to CPU if CUDA requested but not available
      return std::make_unique<CpuClusterBackend>();

    case ClusterBackendType::Auto:
    default:
#ifdef ADAPTIVE_HAS_CUDA
      if (cuda_available()) {
        return std::make_unique<CudaClusterBackend>();
      }
#endif
      return std::make_unique<CpuClusterBackend>();
  }
}

}  // namespace adaptive

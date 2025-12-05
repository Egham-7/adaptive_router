#include "cluster.hpp"

ClusterEngine::ClusterEngine()
    : backend_(adaptive::create_cluster_backend(adaptive::ClusterBackendType::Auto)) {}

ClusterEngine::ClusterEngine(adaptive::ClusterBackendType backend_type)
    : backend_(adaptive::create_cluster_backend(backend_type)) {}

void ClusterEngine::load_centroids(const EmbeddingMatrix& centers) {
  dim_ = static_cast<int>(centers.cols());
  int n_clusters = static_cast<int>(centers.rows());

  // Eigen stores in column-major by default, backend expects row-major
  // Use eval() to ensure contiguous row-major data
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major = centers;
  backend_->load_centroids(row_major.data(), n_clusters, dim_);
}

std::pair<int, float> ClusterEngine::assign(const EmbeddingVector& embedding) {
  return backend_->assign(embedding.data(), static_cast<int>(embedding.size()));
}

int ClusterEngine::get_n_clusters() const noexcept { return backend_->get_n_clusters(); }

bool ClusterEngine::is_gpu_accelerated() const noexcept { return backend_->is_gpu_accelerated(); }

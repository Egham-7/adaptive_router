#pragma once
#include <string>
#include <variant>
#include <vector>

#include "cluster.hpp"
#include "scorer.hpp"

struct ClusteringConfig {
  int max_iter = 300;
  int random_state = 42;
  int n_init = 10;
  std::string algorithm = "lloyd";
  std::string normalization_strategy = "l2";
};

struct RoutingConfig {
  float lambda_min = 0.0f;
  float lambda_max = 2.0f;
  float default_cost_preference = 0.5f;
  int max_alternatives = 5;
};

struct ProfileMetadata {
  int n_clusters;
  std::string embedding_model;
  std::string dtype = "float32";  // "float32" or "float64"
  float silhouette_score;
  ClusteringConfig clustering;
  RoutingConfig routing;
};

// Cluster centers can be either float or double
using ClusterCenters = std::variant<EmbeddingMatrixT<float>, EmbeddingMatrixT<double>>;

struct RouterProfile {
  ClusterCenters cluster_centers;
  std::vector<ModelFeatures> models;
  ProfileMetadata metadata;

  // Load from JSON file
  [[nodiscard]] static RouterProfile from_json(const std::string& path);

  // Load from MessagePack binary file
  [[nodiscard]] static RouterProfile from_binary(const std::string& path);

  // Load from JSON string
  [[nodiscard]] static RouterProfile from_json_string(const std::string& json_str);

  // Type helpers
  [[nodiscard]] bool is_float32() const {
    return std::holds_alternative<EmbeddingMatrixT<float>>(cluster_centers);
  }

  [[nodiscard]] bool is_float64() const {
    return std::holds_alternative<EmbeddingMatrixT<double>>(cluster_centers);
  }

  template<typename Scalar>
  [[nodiscard]] const EmbeddingMatrixT<Scalar>& centers() const {
    return std::get<EmbeddingMatrixT<Scalar>>(cluster_centers);
  }
};

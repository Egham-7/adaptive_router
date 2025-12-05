"""
General tests for Python bindings (adaptive_core_ext)
Tests basic functionality of the NumPy-based API including single and batch routing.
"""

import numpy as np
import pytest

from adaptive_core_ext import Router, RouteResponse


@pytest.fixture
def mock_router_data():
    """Mock data for creating a test router profile"""
    # This would normally come from a real profile file
    # For testing, you'd create a minimal valid profile
    return {
        "metadata": {
            "n_clusters": 3,
            "embedding_model": "all-MiniLM-L6-v2",
            "silhouette_score": 0.5,
        },
        "cluster_centers": np.random.randn(3, 384).tolist(),
        "models": [
            {
                "model_id": "openai/gpt-4",
                "provider": "openai",
                "model_name": "gpt-4",
                "error_rates": [0.05, 0.06, 0.07],
                "cost_per_1m_input_tokens": 30.0,
                "cost_per_1m_output_tokens": 60.0,
            },
            {
                "model_id": "anthropic/claude-3-sonnet",
                "provider": "anthropic",
                "model_name": "claude-3-sonnet-20240229",
                "error_rates": [0.06, 0.05, 0.08],
                "cost_per_1m_input_tokens": 15.0,
                "cost_per_1m_output_tokens": 75.0,
            },
        ],
    }


class TestRouterCreation:
    """Test router factory methods"""

    def test_from_file_requires_valid_path(self):
        """from_file should handle invalid paths"""
        with pytest.raises(Exception):  # Could be FileNotFoundError or RuntimeError
            Router.from_file("nonexistent_profile.json")

    def test_from_json_string_requires_valid_json(self):
        """from_json_string should handle invalid JSON"""
        with pytest.raises(Exception):
            Router.from_json_string("invalid json")

    def test_from_binary_requires_valid_path(self):
        """from_binary should handle invalid paths"""
        with pytest.raises(Exception):
            Router.from_binary("nonexistent_profile.msgpack")


class TestSingleRouting:
    """Test single embedding routing"""

    @pytest.fixture
    def router(self, tmp_path, mock_router_data):
        """Create a test router from mock data"""
        import json

        profile_path = tmp_path / "test_profile.json"
        profile_path.write_text(json.dumps(mock_router_data))

        return Router.from_file(str(profile_path))

    def test_route_with_float32_numpy_array(self, router):
        """Test routing with float32 numpy array"""
        embedding = np.random.randn(384).astype(np.float32)

        response = router.route(embedding, cost_bias=0.5)

        assert isinstance(response, RouteResponse)
        assert isinstance(response.selected_model, str)
        assert len(response.selected_model) > 0
        assert isinstance(response.alternatives, list)
        assert isinstance(response.cluster_id, int)
        assert response.cluster_id >= 0
        assert isinstance(response.cluster_distance, float)
        assert response.cluster_distance >= 0

    def test_route_with_float64_numpy_array(self, router):
        """Test routing with float64 numpy array"""
        embedding = np.random.randn(384)  # Default is float64

        response = router.route(embedding, cost_bias=0.5)

        assert isinstance(response, RouteResponse)
        assert response.selected_model

    def test_route_with_different_cost_bias(self, router):
        """Test routing with different cost_bias values"""
        embedding = np.random.randn(384).astype(np.float32)

        # Test different cost preferences
        response_low_cost = router.route(embedding, cost_bias=1.0)
        response_high_quality = router.route(embedding, cost_bias=0.0)
        response_balanced = router.route(embedding, cost_bias=0.5)

        # All should return valid responses
        assert response_low_cost.selected_model
        assert response_high_quality.selected_model
        assert response_balanced.selected_model

    def test_route_with_wrong_dimension(self, router):
        """Test that routing fails with wrong embedding dimension"""
        wrong_embedding = np.random.randn(512).astype(np.float32)

        with pytest.raises(Exception):
            router.route(wrong_embedding)


class TestBatchRouting:
    """Test batch embedding routing"""

    @pytest.fixture
    def router(self, tmp_path, mock_router_data):
        """Create a test router from mock data"""
        import json

        profile_path = tmp_path / "test_profile.json"
        profile_path.write_text(json.dumps(mock_router_data))

        return Router.from_file(str(profile_path))

    def test_route_batch_with_float32(self, router):
        """Test batch routing with float32 arrays (zero-copy path)"""
        embeddings = np.random.randn(10, 384).astype(np.float32)

        responses = router.route_batch(embeddings, cost_bias=0.5)

        assert isinstance(responses, list)
        assert len(responses) == 10

        for response in responses:
            assert isinstance(response, RouteResponse)
            assert response.selected_model
            assert response.cluster_id >= 0

    def test_route_batch_with_float64(self, router):
        """Test batch routing with float64 arrays (conversion path)"""
        embeddings = np.random.randn(5, 384)  # Default float64

        responses = router.route_batch(embeddings)

        assert len(responses) == 5
        assert all(r.selected_model for r in responses)

    def test_route_batch_empty(self, router):
        """Test batch routing with empty array"""
        embeddings = np.zeros((0, 384), dtype=np.float32)

        responses = router.route_batch(embeddings)

        assert isinstance(responses, list)
        assert len(responses) == 0

    def test_route_batch_single_embedding(self, router):
        """Test batch routing with single embedding"""
        embeddings = np.random.randn(1, 384).astype(np.float32)

        responses = router.route_batch(embeddings)

        assert len(responses) == 1
        assert responses[0].selected_model

    def test_route_batch_wrong_dimension(self, router):
        """Test batch routing with wrong embedding dimension"""
        wrong_embeddings = np.random.randn(5, 512).astype(np.float32)

        with pytest.raises(Exception, match="dimension mismatch"):
            router.route_batch(wrong_embeddings)

    def test_route_batch_consistency(self, router):
        """Test that batch routing gives same results as individual routing"""
        embeddings = np.random.randn(5, 384).astype(np.float32)

        # Batch route
        batch_responses = router.route_batch(embeddings, cost_bias=0.5)

        # Individual routes
        individual_responses = []
        for i in range(5):
            response = router.route(embeddings[i], cost_bias=0.5)
            individual_responses.append(response)

        # Results should match
        for batch_resp, indiv_resp in zip(batch_responses, individual_responses):
            assert batch_resp.selected_model == indiv_resp.selected_model
            assert batch_resp.cluster_id == indiv_resp.cluster_id
            assert abs(batch_resp.cluster_distance - indiv_resp.cluster_distance) < 1e-5


class TestRouterIntrospection:
    """Test router introspection methods"""

    @pytest.fixture
    def router(self, tmp_path, mock_router_data):
        """Create a test router from mock data"""
        import json

        profile_path = tmp_path / "test_profile.json"
        profile_path.write_text(json.dumps(mock_router_data))

        return Router.from_file(str(profile_path))

    def test_get_supported_models(self, router):
        """Test get_supported_models returns list of model IDs"""
        models = router.get_supported_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_get_n_clusters(self, router):
        """Test get_n_clusters returns cluster count"""
        n_clusters = router.get_n_clusters()

        assert isinstance(n_clusters, int)
        assert n_clusters > 0
        assert n_clusters == 3  # From mock data

    def test_get_embedding_dim(self, router):
        """Test get_embedding_dim returns dimension"""
        dim = router.get_embedding_dim()

        assert isinstance(dim, int)
        assert dim == 384  # Standard for all-MiniLM-L6-v2

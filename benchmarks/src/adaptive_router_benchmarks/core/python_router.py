"""Pure Python implementation of profile router using Cactus embeddings.

This router uses the actual Cactus LFM2-350M model for embeddings on Mac/ARM,
with an automatic fallback to BAAI/bge-large-en-v1.5 for x86 testing.
"""

import json
import platform
import numpy as np
from pathlib import Path
from typing import Any, Optional


class PythonProfileRouter:
    """Pure Python profile router using Cactus embeddings.

    This router automatically detects the platform and uses:
    - Real Cactus LFM2-350M embeddings on Mac/ARM (production)
    - BAAI/bge-large-en-v1.5 fallback on x86 (testing only)

    For accurate routing results, you MUST run this on Mac with Cactus built.
    """

    def __init__(
        self,
        profile_path: str | Path,
        cactus_model_path: Optional[str] = None,
        cactus_lib_path: Optional[str] = None,
        force_mock: bool = False
    ):
        """Initialize the router with profile and embedding model.

        Args:
            profile_path: Path to production_profile.json
            cactus_model_path: Path to Cactus LFM2-350M model directory
                              (default: auto-detect from repo structure)
            cactus_lib_path: Optional path to libcactus library
            force_mock: Force use of mock embeddings even on ARM
        """
        self.profile_path = Path(profile_path)
        self.force_mock = force_mock

        # Load profile
        with open(self.profile_path) as f:
            self.profile_data = json.load(f)

        # Extract data
        self.cluster_centers = np.array(
            self.profile_data["cluster_centers"]["cluster_centers"]
        )
        self.models = {m["model_id"]: m for m in self.profile_data["models"]}

        # Auto-detect Cactus model path if not provided
        if cactus_model_path is None:
            # Default: ../../../cactus/weights/lfm2-350m from this file
            repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            cactus_model_path = str(repo_root / "cactus" / "weights" / "lfm2-350m")

        self.cactus_model_path = cactus_model_path
        self.cactus_lib_path = cactus_lib_path

        # Initialize embedding model (Cactus or fallback)
        self._init_embedding_model()

    def _init_embedding_model(self):
        """Initialize embedding model (Cactus on ARM, fallback on x86)."""

        machine = platform.machine().lower()
        is_arm = machine in ['arm64', 'aarch64']

        # Try to use Cactus on ARM
        if is_arm and not self.force_mock:
            try:
                from adaptive_router_benchmarks.bindings import CactusModel, CactusNotAvailableError

                print("\n🌵 Initializing Cactus LFM2-350M for embeddings")
                print(f"   Model path: {self.cactus_model_path}")

                self.cactus_model = CactusModel(
                    self.cactus_model_path,
                    context_size=2048,
                    lib_path=self.cactus_lib_path
                )

                # Test embedding dimension
                test_emb = self.cactus_model.embed("test")
                print(f"   Embedding dimension: {len(test_emb)}")
                print("✅ Using REAL Cactus embeddings (production mode)\n")

                self.use_cactus = True
                self.encoder = None
                return

            except Exception as e:
                print(f"\n⚠️  Failed to initialize Cactus: {e}")
                print("   Falling back to BAAI/bge-large-en-v1.5...\n")

        # Fallback to sentence-transformers
        if not is_arm:
            print(f"\n⚠️  Platform: {machine} (x86)")
            print("   Cactus requires ARM architecture")

        print("   Using BAAI/bge-large-en-v1.5 as fallback (1024d)")
        print("   ⚠️  WARNING: This is NOT identical to Cactus embeddings!")
        print("   ⚠️  For accurate results, run on Mac with Cactus built.\n")

        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(
            "BAAI/bge-large-en-v1.5",
            device="cpu"
        )
        self.use_cactus = False
        self.cactus_model = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Cactus or fallback.

        Args:
            text: Input text

        Returns:
            Embedding vector (1024 dimensions)
        """
        if self.use_cactus:
            # Use real Cactus embeddings
            embedding = self.cactus_model.embed(text)
            # Normalize (Cactus outputs raw embeddings)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        else:
            # Use BAAI/bge fallback
            embedding = self.encoder.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding

    def route(self, prompt: str, cost_bias: float = 0.5) -> dict[str, Any]:
        """Route prompt to best model based on profile.

        Args:
            prompt: User prompt
            cost_bias: 0.0 = prefer speed (small models)
                      1.0 = prefer quality (large models)

        Returns:
            Dict with:
                - selected_model: Model ID
                - cluster: Cluster ID
                - routing_time_ms: Time to compute routing
                - model_size_mb: Selected model size
                - avg_tokens_per_sec: Expected throughput
                - error_rate: Expected error rate
                - alternatives: Empty list
        """
        import time

        start = time.perf_counter()

        # 1. Encode prompt (1024 dimensions)
        embedding = self._get_embedding(prompt)

        # 2. Find nearest cluster
        distances = np.linalg.norm(self.cluster_centers - embedding, axis=1)
        cluster_id = int(np.argmin(distances))

        # 3. Score models for this cluster
        lambda_value = cost_bias * 2.0  # Map [0, 1] to [0, 2]

        best_model = None
        best_score = float('inf')

        for model_id, model in self.models.items():
            # Get error rate for this cluster
            error_rates = model.get("error_rates", [])
            if cluster_id < len(error_rates):
                error_rate = error_rates[cluster_id]
            else:
                error_rate = np.mean(error_rates) if error_rates else 0.1

            # Normalize cost (model size as proxy)
            size_mb = model.get("size_mb", 500)
            normalized_cost = size_mb / 2000.0  # Normalize to ~[0, 1]

            # Score: error_rate + lambda * normalized_cost
            score = error_rate + (lambda_value * normalized_cost)

            if score < best_score:
                best_score = score
                best_model = model_id

        routing_time_ms = (time.perf_counter() - start) * 1000

        # Get model metadata
        model_meta = self.models[best_model]

        return {
            "selected_model": best_model,
            "cluster": cluster_id,
            "routing_time_ms": routing_time_ms,
            "model_size_mb": model_meta.get("size_mb", 0),
            "avg_tokens_per_sec": model_meta.get("avg_tokens_per_sec", 0),
            "error_rate": model_meta.get("error_rates", [])[cluster_id]
                if cluster_id < len(model_meta.get("error_rates", []))
                else np.mean(model_meta.get("error_rates", [0.1])),
            "alternatives": [],
        }

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get metadata for a model."""
        return self.models.get(model_id, {})

    def __del__(self):
        """Cleanup Cactus model on deletion."""
        if self.use_cactus and self.cactus_model is not None:
            try:
                self.cactus_model.destroy()
            except:
                pass

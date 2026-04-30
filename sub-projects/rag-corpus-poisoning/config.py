"""Configuration dataclass for the RAG Corpus Poisoning Detector."""

from dataclasses import dataclass, field


@dataclass
class DetectorConfig:
    """All thresholds and settings for the temporal corpus drift detector.

    Attributes:
        theta_corpus: JSD drift threshold above which an alert is raised.
        delta_fast: Centroid L2 distance fast-check threshold (Stage 1).
        projection_dims: Target dimensionality for random projection before KDE/JSD.
        window_size: Number of historical snapshots in the sliding window.
        hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN anomaly detection.
        sigma_similarity: Cosine similarity threshold for provenance graph edges.
        embedding_model: Primary sentence-transformer model name.
        fallback_model: Fallback model when the primary is unavailable.
        qdrant_host: Hostname for the Qdrant vector store.
        qdrant_port: Port for the Qdrant vector store.
        redis_host: Hostname for the Redis cache.
        redis_port: Port for the Redis cache.
        redis_ttl: TTL in seconds for cached centroids and covariances.
        max_candidate_docs: Maximum documents evaluated in counterfactual analysis.
        consecutive_windows_for_escalation: Windows with drift before escalating severity.
        severity_p1_threshold: JSD score above which alert is P1 (critical).
        severity_p2_threshold: JSD score above which alert is P2 (high).
    """

    # Detection thresholds
    theta_corpus: float = 0.15
    delta_fast: float = 0.05
    projection_dims: int = 32
    window_size: int = 4
    hdbscan_min_cluster_size: int = 3
    sigma_similarity: float = 0.85

    # Model selection
    embedding_model: str = "hkunlp/instructor-xl"
    fallback_model: str = "all-mpnet-base-v2"

    # External service coordinates
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl: int = 86400  # 24 hours

    # Provenance and escalation
    max_candidate_docs: int = 50
    consecutive_windows_for_escalation: int = 2

    # Alert severity thresholds
    severity_p1_threshold: float = 0.30
    severity_p2_threshold: float = 0.20
    # severity_p3 starts at theta_corpus (0.15)

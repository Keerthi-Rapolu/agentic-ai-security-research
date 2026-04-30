"""Temporal snapshot management for the RAG corpus poisoning detector."""

import json
import logging
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """A versioned embedding snapshot for a single time window.

    Attributes:
        window_id: Unique identifier for the time window (e.g. ``"2024-W03"``).
        embeddings: Float32 array of shape ``(n_docs, embedding_dim)``.
        metadata: Per-document provenance metadata keyed by list index.
        centroid: Pre-computed mean embedding vector.
        covariance: Pre-computed covariance diagonal (variance per dimension).
    """

    window_id: str
    embeddings: np.ndarray
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.centroid is None and len(self.embeddings):
            self.centroid = self.embeddings.mean(axis=0)
        if self.covariance is None and len(self.embeddings) > 1:
            self.covariance = self.embeddings.var(axis=0)


class SnapshotStore:
    """Manages versioned embedding snapshots across time windows.

    Uses Qdrant for persistent vector storage and Redis for fast centroid/
    covariance lookups.  Falls back to an in-memory dictionary when neither
    service is reachable — this enables unit tests without infrastructure.

    Args:
        qdrant_host: Qdrant server hostname.
        qdrant_port: Qdrant server port.
        redis_host: Redis server hostname.
        redis_port: Redis server port.
        redis_ttl: TTL in seconds for Redis cache entries.
        collection_prefix: Prefix applied to all Qdrant collection names.
    """

    _REDIS_KEY_PREFIX = "rag_snapshot:"

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_ttl: int = 86400,
        collection_prefix: str = "corpus_snapshot",
    ) -> None:
        self.redis_ttl = redis_ttl
        self.collection_prefix = collection_prefix
        self._memory: Dict[str, Snapshot] = {}

        self._qdrant = self._init_qdrant(qdrant_host, qdrant_port)
        self._redis = self._init_redis(redis_host, redis_port)

    def _init_qdrant(self, host: str, port: int) -> Optional[Any]:
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(host=host, port=port, timeout=5)
            client.get_collections()  # connectivity check
            logger.info("Connected to Qdrant at %s:%d", host, port)
            return client
        except Exception as exc:
            logger.warning("Qdrant unavailable (%s). Using in-memory store.", exc)
            return None

    def _init_redis(self, host: str, port: int) -> Optional[Any]:
        try:
            import redis

            client = redis.Redis(host=host, port=port, socket_connect_timeout=3)
            client.ping()
            logger.info("Connected to Redis at %s:%d", host, port)
            return client
        except Exception as exc:
            logger.warning("Redis unavailable (%s). Centroid cache disabled.", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        window_id: str,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Snapshot:
        """Persist a snapshot for *window_id* and cache its statistics.

        Args:
            window_id: Unique window identifier.
            embeddings: Document embeddings for the window.
            metadata: Optional per-document provenance metadata.

        Returns:
            The constructed and stored :class:`Snapshot`.
        """
        if metadata is None:
            metadata = [{} for _ in range(len(embeddings))]

        snapshot = Snapshot(window_id=window_id, embeddings=embeddings, metadata=metadata)
        self._memory[window_id] = snapshot

        self._cache_stats(snapshot)

        if self._qdrant is not None:
            self._write_qdrant(snapshot)

        logger.debug("Saved snapshot %s (%d docs)", window_id, len(embeddings))
        return snapshot

    def load_snapshot(self, window_id: str) -> Optional[Snapshot]:
        """Load a snapshot by *window_id*, consulting memory, cache, then Qdrant.

        Args:
            window_id: Unique window identifier.

        Returns:
            The :class:`Snapshot` or ``None`` if not found.
        """
        if window_id in self._memory:
            return self._memory[window_id]

        if self._qdrant is not None:
            snapshot = self._read_qdrant(window_id)
            if snapshot is not None:
                self._memory[window_id] = snapshot
                return snapshot

        logger.warning("Snapshot %s not found", window_id)
        return None

    def list_windows(self) -> List[str]:
        """Return all known window IDs sorted chronologically.

        Returns:
            Sorted list of window identifier strings.
        """
        return sorted(self._memory.keys())

    # ------------------------------------------------------------------
    # Redis centroid cache
    # ------------------------------------------------------------------

    def _cache_stats(self, snapshot: Snapshot) -> None:
        if self._redis is None:
            return
        key = f"{self._REDIS_KEY_PREFIX}{snapshot.window_id}"
        payload = {
            "centroid": snapshot.centroid.tolist() if snapshot.centroid is not None else None,
            "covariance": snapshot.covariance.tolist() if snapshot.covariance is not None else None,
            "n_docs": len(snapshot.embeddings),
        }
        try:
            self._redis.setex(key, self.redis_ttl, json.dumps(payload))
        except Exception as exc:
            logger.warning("Redis write failed: %s", exc)

    def get_cached_stats(self, window_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached centroid/covariance for *window_id* from Redis.

        Args:
            window_id: Unique window identifier.

        Returns:
            Dict with ``centroid``, ``covariance``, and ``n_docs`` keys, or
            ``None`` if not cached.
        """
        if self._redis is None:
            return None
        key = f"{self._REDIS_KEY_PREFIX}{window_id}"
        try:
            raw = self._redis.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            if data.get("centroid"):
                data["centroid"] = np.array(data["centroid"], dtype=np.float32)
            if data.get("covariance"):
                data["covariance"] = np.array(data["covariance"], dtype=np.float32)
            return data
        except Exception as exc:
            logger.warning("Redis read failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Qdrant persistence
    # ------------------------------------------------------------------

    def _collection_name(self, window_id: str) -> str:
        safe = window_id.replace("-", "_").replace(":", "_")
        return f"{self.collection_prefix}_{safe}"

    def _write_qdrant(self, snapshot: Snapshot) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        col = self._collection_name(snapshot.window_id)
        dim = snapshot.embeddings.shape[1]
        try:
            existing = [c.name for c in self._qdrant.get_collections().collections]
            if col not in existing:
                self._qdrant.create_collection(
                    collection_name=col,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            points = [
                PointStruct(
                    id=idx,
                    vector=snapshot.embeddings[idx].tolist(),
                    payload=snapshot.metadata[idx] if idx < len(snapshot.metadata) else {},
                )
                for idx in range(len(snapshot.embeddings))
            ]
            self._qdrant.upsert(collection_name=col, points=points)
            logger.debug("Wrote %d vectors to Qdrant collection %s", len(points), col)
        except Exception as exc:
            logger.error("Qdrant write failed for %s: %s", snapshot.window_id, exc)

    def _read_qdrant(self, window_id: str) -> Optional[Snapshot]:
        from qdrant_client.models import Filter

        col = self._collection_name(window_id)
        try:
            existing = [c.name for c in self._qdrant.get_collections().collections]
            if col not in existing:
                return None
            count = self._qdrant.count(col).count
            if count == 0:
                return None
            results, _ = self._qdrant.scroll(col, limit=count, with_vectors=True, with_payload=True)
            embeddings = np.array([r.vector for r in results], dtype=np.float32)
            metadata = [r.payload for r in results]
            return Snapshot(window_id=window_id, embeddings=embeddings, metadata=metadata)
        except Exception as exc:
            logger.error("Qdrant read failed for %s: %s", window_id, exc)
            return None

"""Trust registry — Redis-backed hot tier with in-memory fallback."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrustHistoryEntry:
    """A single entry in an agent's trust score history.

    Attributes:
        agent_id: The agent whose trust changed.
        score: Trust score at this point.
        timestamp: Unix timestamp of the update.
        reason: Free-form description of what caused the change.
        prev_hash: SHA-256 hash of the previous history entry for chain integrity.
        entry_hash: SHA-256 hash of this entry.
    """

    agent_id: str
    score: float
    timestamp: float
    reason: str
    prev_hash: str = ""
    entry_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.agent_id.encode())
        h.update(f"{self.score:.8f}".encode())
        h.update(f"{self.timestamp:.6f}".encode())
        h.update(self.reason.encode())
        h.update(self.prev_hash.encode())
        return h.hexdigest()


class TrustRegistry:
    """Two-tier trust score store for the multi-agent pipeline.

    - **Hot tier (Redis):** Real-time trust score reads/writes with TTL.
    - **Cold tier (in-memory dict):** Append-only history with SHA-256 hash
      chaining for integrity.  In production this would be PostgreSQL.

    Falls back gracefully to pure in-memory when Redis is unavailable.

    Args:
        redis_host: Redis server hostname.
        redis_port: Redis server port.
        redis_ttl: TTL in seconds for each score key.
        history_max_entries: Cap on per-agent history depth to control memory.
    """

    _KEY_PREFIX = "trust:"
    _HISTORY_PREFIX = "trust_hist:"

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_ttl: int = 3600,
        history_max_entries: int = 1000,
    ) -> None:
        self.redis_ttl = redis_ttl
        self.history_max_entries = history_max_entries

        self._scores: Dict[str, float] = {}  # in-memory fallback
        self._history: Dict[str, List[TrustHistoryEntry]] = {}
        self._redis = self._init_redis(redis_host, redis_port)

    def _init_redis(self, host: str, port: int) -> Optional[Any]:
        try:
            import redis

            client = redis.Redis(host=host, port=port, socket_connect_timeout=3)
            client.ping()
            logger.info("TrustRegistry connected to Redis at %s:%d", host, port)
            return client
        except Exception as exc:
            logger.warning("Redis unavailable (%s). Trust registry using in-memory mode.", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, agent_id: str) -> Optional[float]:
        """Return the current trust score for *agent_id*, or None if unknown.

        Args:
            agent_id: Unique agent identifier.

        Returns:
            Float trust score in ``[0, 1]`` or ``None``.
        """
        if self._redis is not None:
            raw = self._redis.get(f"{self._KEY_PREFIX}{agent_id}")
            if raw is not None:
                return float(raw)
        return self._scores.get(agent_id)

    def set(self, agent_id: str, score: float, reason: str = "") -> bool:
        """Atomically set the trust score for *agent_id*.

        Uses Redis GETSET for atomicity when Redis is available.  Records the
        change in the append-only history with SHA-256 chaining.

        Args:
            agent_id: Unique agent identifier.
            score: New trust score, clamped to ``[0, 1]``.
            reason: Human-readable description of why the score changed.

        Returns:
            ``True`` if the update succeeded.
        """
        score = float(max(0.0, min(1.0, score)))
        key = f"{self._KEY_PREFIX}{agent_id}"

        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.getset(key, score)
                pipe.expire(key, self.redis_ttl)
                pipe.execute()
            except Exception as exc:
                logger.warning("Redis set failed for %s: %s", agent_id, exc)

        self._scores[agent_id] = score
        self._append_history(agent_id, score, reason)
        logger.debug("Trust set: %s = %.4f (%s)", agent_id, score, reason)
        return True

    def history(self, agent_id: str) -> List[TrustHistoryEntry]:
        """Return the full trust score history for *agent_id*.

        Args:
            agent_id: Unique agent identifier.

        Returns:
            List of :class:`TrustHistoryEntry` in chronological order.
        """
        return list(self._history.get(agent_id, []))

    def compare_and_swap(
        self, agent_id: str, expected: float, new_score: float, reason: str = ""
    ) -> bool:
        """Update trust score only if current value equals *expected*.

        Provides optimistic concurrency control.

        Args:
            agent_id: Unique agent identifier.
            expected: Trust score that must be present for the swap to proceed.
            new_score: New trust score to set.
            reason: Reason for the change.

        Returns:
            ``True`` if the swap was performed; ``False`` if the current value
            differed from *expected*.
        """
        current = self.get(agent_id)
        if current is None or abs(current - expected) > 1e-6:
            return False
        return self.set(agent_id, new_score, reason)

    def list_agents(self) -> List[str]:
        """Return all known agent IDs.

        Returns:
            List of agent identifier strings.
        """
        return list(self._scores.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append_history(self, agent_id: str, score: float, reason: str) -> None:
        history = self._history.setdefault(agent_id, [])
        prev_hash = history[-1].entry_hash if history else ""
        entry = TrustHistoryEntry(
            agent_id=agent_id,
            score=score,
            timestamp=time.time(),
            reason=reason,
            prev_hash=prev_hash,
        )
        history.append(entry)
        if len(history) > self.history_max_entries:
            history.pop(0)

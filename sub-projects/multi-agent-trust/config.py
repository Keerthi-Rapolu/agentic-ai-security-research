"""Configuration for the multi-agent trust propagation engine."""

from dataclasses import dataclass, field
from typing import Dict


# Default initial trust scores by agent role
_DEFAULT_ROLE_PRIORS: Dict[str, float] = {
    "orchestrator": 0.95,
    "retrieval": 0.80,
    "code_execution": 0.70,
    "external_api": 0.65,
    "user_facing": 0.75,
    "unverified_third_party": 0.40,
    "default": 0.60,
}


@dataclass
class TrustConfig:
    """All thresholds and settings for the multi-agent trust engine.

    Attributes:
        tau_c: Byzantine compromise threshold.  Agents with tau < tau_c are
            flagged as Byzantine.
        tau_eff_c: Effective trust threshold for flagging downstream outputs.
        alpha: Trust degradation exponent in f(tau) = tau^alpha.
        half_life_hours: Half-life of trust recovery in hours.
        alignment_penalty: Trust decrement per alignment-violation flag.
        corpus_penalty: Trust decrement per corpus-poisoning dependency flag.
        anomaly_penalty_factor: Per-unit anomaly score trust decrement.
        positive_reinforcement: Trust increment per validated output.
        role_priors: Initial trust scores keyed by agent role string.
        redis_host: Redis server hostname.
        redis_port: Redis server port.
        redis_ttl: TTL in seconds for Redis trust score entries.
        history_max_entries: Maximum history entries per agent in the in-memory
            cold tier.
    """

    tau_c: float = 0.30
    tau_eff_c: float = 0.25
    alpha: float = 1.0
    half_life_hours: float = 24.0
    alignment_penalty: float = 0.15
    corpus_penalty: float = 0.10
    anomaly_penalty_factor: float = 0.05
    positive_reinforcement: float = 0.02
    role_priors: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_ROLE_PRIORS))
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl: int = 3600
    history_max_entries: int = 1000

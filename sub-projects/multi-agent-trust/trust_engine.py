"""Trust score computation and propagation for multi-agent LLM pipelines."""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

from config import TrustConfig
from dependency_graph import AgentDependencyGraph
from registry import TrustRegistry

logger = logging.getLogger(__name__)


class TrustEngine:
    """Computes, propagates, and updates agent trust scores.

    Implements the trust propagation algebra:

        τ_j_eff = τ_j × ∏(f(τ_i_eff) for i in parents(j))
        f(τ) = τ^α

    Trust scores recover exponentially toward the role prior:

        τ(t) = τ_0 + (τ_degraded − τ_0) × exp(−λ (t − t_event))

    An agent is Byzantine if its intrinsic trust τ < τ_c.

    Args:
        registry: :class:`TrustRegistry` for persisting trust scores.
        graph: :class:`AgentDependencyGraph` representing agent dependencies.
        config: :class:`TrustConfig` with all thresholds and settings.
    """

    def __init__(
        self,
        registry: TrustRegistry,
        graph: AgentDependencyGraph,
        config: Optional[TrustConfig] = None,
    ) -> None:
        self.registry = registry
        self.graph = graph
        self.config = config or TrustConfig()

        # Track last degradation event times for recovery calculation
        self._last_event_time: Dict[str, float] = {}
        # Track the role prior for each agent for recovery target
        self._role_priors: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        role: str = "default",
        initial_trust: Optional[float] = None,
    ) -> float:
        """Register an agent with its role prior trust score.

        Args:
            agent_id: Unique agent identifier.
            role: Agent role string for prior lookup.
            initial_trust: Override the role-based prior.

        Returns:
            The initial trust score assigned to the agent.
        """
        prior = initial_trust if initial_trust is not None else (
            self.config.role_priors.get(role, self.config.role_priors.get("default", 0.60))
        )
        self._role_priors[agent_id] = prior
        if not self.graph.has_agent(agent_id):
            self.graph.add_agent(agent_id, role=role, initial_trust=prior)
        self.registry.set(agent_id, prior, reason=f"initial registration as {role}")
        logger.info("Registered agent %s (role=%s, τ₀=%.3f)", agent_id, role, prior)
        return prior

    # ------------------------------------------------------------------
    # Trust retrieval and computation
    # ------------------------------------------------------------------

    def get_trust(self, agent_id: str) -> float:
        """Return the current intrinsic trust score for *agent_id*, with recovery.

        Applies exponential recovery toward the role prior if time has passed
        since the last degradation event.

        Args:
            agent_id: Unique agent identifier.

        Returns:
            Current trust score in ``[0, 1]``.
        """
        raw = self.registry.get(agent_id)
        if raw is None:
            logger.warning("Agent %s not registered; returning 0.0", agent_id)
            return 0.0

        last_event = self._last_event_time.get(agent_id)
        tau_0 = self._role_priors.get(agent_id, raw)
        if last_event is None or raw >= tau_0:
            return float(raw)

        # Exponential recovery: τ(t) = τ_0 + (τ_deg − τ_0) × exp(−λΔt)
        lam = math.log(2) / (self.config.half_life_hours * 3600)
        dt = time.time() - last_event
        recovered = tau_0 + (raw - tau_0) * math.exp(-lam * dt)
        recovered = max(0.0, min(1.0, recovered))
        return float(recovered)

    def compute_effective_trust(self, agent_id: str) -> float:
        """Compute the effective trust of *agent_id* accounting for all ancestors.

        τ_eff = τ_j × ∏(f(τ_i_eff) for i in parents(j))
        where f(τ) = τ^α

        This is computed recursively (memoised per call to avoid exponential
        traversal).

        Args:
            agent_id: Agent whose effective trust is needed.

        Returns:
            Effective trust score in ``[0, 1]``.
        """
        cache: Dict[str, float] = {}
        return self._eff_trust_recursive(agent_id, cache)

    def _eff_trust_recursive(self, agent_id: str, cache: Dict[str, float]) -> float:
        if agent_id in cache:
            return cache[agent_id]

        tau_j = self.get_trust(agent_id)
        parents = self.graph.get_parents(agent_id)

        if not parents:
            cache[agent_id] = tau_j
            return tau_j

        product = 1.0
        for parent in parents:
            tau_parent_eff = self._eff_trust_recursive(parent, cache)
            product *= tau_parent_eff ** self.config.alpha

        tau_eff = tau_j * product
        tau_eff = max(0.0, min(1.0, tau_eff))
        cache[agent_id] = tau_eff
        return tau_eff

    def is_byzantine(self, agent_id: str) -> bool:
        """Return True if *agent_id*'s intrinsic trust is below τ_c.

        Args:
            agent_id: Agent ID to check.

        Returns:
            Boolean.
        """
        return self.get_trust(agent_id) < self.config.tau_c

    # ------------------------------------------------------------------
    # Trust updates
    # ------------------------------------------------------------------

    def update_trust(
        self, agent_id: str, delta: float, reason: str = ""
    ) -> Tuple[float, float]:
        """Apply a trust score delta and return (old_score, new_score).

        The score is clamped to ``[0, 1]``.  The last degradation event
        timestamp is updated when delta < 0 so that recovery starts from
        the new value.

        Args:
            agent_id: Agent ID.
            delta: Amount to add to the current trust score (negative = penalty).
            reason: Human-readable reason for the update.

        Returns:
            Tuple of ``(old_score, new_score)``.
        """
        old = self.registry.get(agent_id)
        if old is None:
            logger.warning("update_trust called for unregistered agent %s", agent_id)
            return (0.0, 0.0)

        new = max(0.0, min(1.0, old + delta))
        self.registry.set(agent_id, new, reason=reason)

        if delta < 0:
            self._last_event_time[agent_id] = time.time()

        if self.is_byzantine(agent_id):
            logger.warning("Agent %s is now Byzantine (τ=%.3f < τ_c=%.3f)", agent_id, new, self.config.tau_c)

        return float(old), float(new)

    def apply_alignment_penalty(self, agent_id: str) -> Tuple[float, float]:
        """Apply the standard alignment-violation trust penalty.

        Args:
            agent_id: Agent ID receiving the penalty.

        Returns:
            Tuple of ``(old_score, new_score)``.
        """
        return self.update_trust(agent_id, -self.config.alignment_penalty, reason="alignment_violation")

    def apply_corpus_penalty(self, agent_id: str) -> Tuple[float, float]:
        """Apply the corpus-poisoning dependency trust penalty.

        Args:
            agent_id: Agent ID receiving the penalty.

        Returns:
            Tuple of ``(old_score, new_score)``.
        """
        return self.update_trust(agent_id, -self.config.corpus_penalty, reason="corpus_poisoning_dependency")

    def apply_positive_reinforcement(self, agent_id: str) -> Tuple[float, float]:
        """Apply a positive reinforcement increment (capped at role prior).

        Args:
            agent_id: Agent ID receiving reinforcement.

        Returns:
            Tuple of ``(old_score, new_score)``.
        """
        tau_0 = self._role_priors.get(agent_id, 1.0)
        current = self.registry.get(agent_id) or 0.0
        increment = min(self.config.positive_reinforcement, tau_0 - current)
        if increment <= 0:
            return float(current), float(current)
        return self.update_trust(agent_id, increment, reason="positive_reinforcement")

    def quarantine(self, agent_id: str) -> None:
        """Immediately set *agent_id*'s trust to 0.0 (Byzantine quarantine).

        Args:
            agent_id: Agent ID to quarantine.
        """
        old = self.registry.get(agent_id) or 0.0
        self.registry.set(agent_id, 0.0, reason="manual_quarantine")
        self._last_event_time[agent_id] = time.time()
        logger.warning("Agent %s quarantined (τ: %.3f → 0.0)", agent_id, old)

    # ------------------------------------------------------------------
    # Pipeline-level analysis
    # ------------------------------------------------------------------

    def get_pipeline_trust_map(self) -> Dict[str, Dict[str, float]]:
        """Return intrinsic and effective trust scores for all registered agents.

        Returns:
            Dict mapping agent ID to ``{"intrinsic": float, "effective": float}``.
        """
        result = {}
        for agent_id in self.registry.list_agents():
            result[agent_id] = {
                "intrinsic": self.get_trust(agent_id),
                "effective": self.compute_effective_trust(agent_id),
            }
        return result

    def get_contaminated_descendants(self, byzantine_agent_id: str) -> List[str]:
        """Return all descendants whose effective trust falls below τ_eff_c.

        Args:
            byzantine_agent_id: Agent declared Byzantine.

        Returns:
            List of descendant agent IDs with contaminated effective trust.
        """
        descendants = self.graph.get_descendants(byzantine_agent_id)
        contaminated = []
        for desc in descendants:
            eff = self.compute_effective_trust(desc)
            if eff < self.config.tau_eff_c:
                contaminated.append(desc)
        return contaminated

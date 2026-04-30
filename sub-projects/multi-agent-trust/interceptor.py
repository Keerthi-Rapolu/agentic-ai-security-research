"""Inter-agent message interceptor for trust-aware pipeline communication."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import TrustConfig
from trust_engine import TrustEngine

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """An inter-agent message in the pipeline.

    Attributes:
        sender_id: Agent ID that produced this message.
        receiver_id: Agent ID that will consume this message.
        content: The raw message payload.
        metadata: Optional metadata (prompt, tool calls, timestamps, etc.).
        trust_envelope: Trust metadata attached by the interceptor.
    """

    sender_id: str
    receiver_id: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    trust_envelope: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterceptResult:
    """Result of intercepting a single inter-agent message.

    Attributes:
        effective_trust: Computed effective trust of the sender at intercept time.
        intrinsic_trust: Sender's intrinsic (non-propagated) trust score.
        is_flagged: Whether the message is flagged due to low effective trust.
        should_quarantine: Whether the engine recommends quarantining the sender.
        warning_tags: List of human-readable warning strings attached to the message.
        message: The original :class:`AgentMessage` with updated trust envelope.
    """

    effective_trust: float
    intrinsic_trust: float
    is_flagged: bool
    should_quarantine: bool
    warning_tags: List[str]
    message: AgentMessage


class MessageInterceptor:
    """Intercepts inter-agent messages and attaches trust metadata.

    For each message from agent A to agent B:
    1. Reads A's current trust score from the registry.
    2. Computes A's effective trust (accounting for upstream dependencies).
    3. Flags the message if effective trust < τ_eff_c.
    4. Recommends quarantine if intrinsic trust < τ_c (Byzantine).
    5. Attaches a trust envelope to the message for downstream agents.

    Args:
        engine: :class:`TrustEngine` for trust computation.
        config: :class:`TrustConfig` with thresholds.
    """

    def __init__(
        self,
        engine: TrustEngine,
        config: Optional[TrustConfig] = None,
    ) -> None:
        self.engine = engine
        self.config = config or TrustConfig()

    def intercept(self, message: AgentMessage) -> InterceptResult:
        """Intercept a message and evaluate its trust status.

        Args:
            message: The :class:`AgentMessage` to evaluate.

        Returns:
            :class:`InterceptResult` with effective trust, flag, and quarantine
            recommendation.
        """
        sender = message.sender_id
        intrinsic = self.engine.get_trust(sender)
        effective = self.engine.compute_effective_trust(sender)

        is_flagged = effective < self.config.tau_eff_c
        should_quarantine = intrinsic < self.config.tau_c

        warning_tags: List[str] = []
        if is_flagged:
            warning_tags.append(f"LOW_EFFECTIVE_TRUST:{effective:.3f}")
        if should_quarantine:
            warning_tags.append(f"BYZANTINE_SENDER:{intrinsic:.3f}")

        ancestors = self.engine.graph.get_ancestors(sender)
        for ancestor in ancestors:
            if self.engine.is_byzantine(ancestor):
                warning_tags.append(f"BYZANTINE_ANCESTOR:{ancestor}")

        message.trust_envelope = {
            "sender_id": sender,
            "intrinsic_trust": intrinsic,
            "effective_trust": effective,
            "is_flagged": is_flagged,
            "should_quarantine": should_quarantine,
            "warning_tags": warning_tags,
        }

        if is_flagged:
            logger.warning(
                "Message from %s to %s FLAGGED: τ_eff=%.3f < τ_eff_c=%.3f | tags=%s",
                sender, message.receiver_id, effective, self.config.tau_eff_c, warning_tags,
            )
        else:
            logger.debug(
                "Message from %s to %s: τ_eff=%.3f OK",
                sender, message.receiver_id, effective,
            )

        return InterceptResult(
            effective_trust=effective,
            intrinsic_trust=intrinsic,
            is_flagged=is_flagged,
            should_quarantine=should_quarantine,
            warning_tags=warning_tags,
            message=message,
        )

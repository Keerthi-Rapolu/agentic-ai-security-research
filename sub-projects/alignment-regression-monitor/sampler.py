"""Stratified output sampling strategy for the alignment regression monitor."""

import logging
import random
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

# Keywords used by the principle-triggered sampler to identify outputs that
# likely touch constitutional dimensions (harmlessness, honesty, helpfulness).
_PRINCIPLE_KEYWORDS = frozenset(
    [
        "should", "must", "cannot", "refuse", "sorry", "unable",
        "harmful", "illegal", "dangerous", "safe", "honest",
        "privacy", "confidential", "sensitive", "policy",
        "help", "assist", "support", "guidance",
    ]
)

# Keywords that suggest an output may have already been flagged by an
# inference-time safety classifier.
_SAFETY_KEYWORDS = frozenset(
    [
        "warning", "flagged", "blocked", "unsafe", "violation",
        "jailbreak", "prohibited", "forbidden", "threat",
    ]
)


class OutputSampler:
    """Decides whether a given production output should be monitored.

    Three sampling tiers are supported:

    - **Random (1%):** Uniform baseline coverage.
    - **Principle-triggered (5%):** Applied when the output text contains
      constitutional-principle-relevant keywords.
    - **Safety-triggered (100%):** Applied when the output text contains
      keywords suggesting an inference-time safety classifier was activated.

    Args:
        random_rate: Fraction of outputs sampled at the random tier.
        principle_rate: Fraction sampled at the principle-triggered tier.
        safety_rate: Fraction sampled at the safety-triggered tier.
        seed: Optional random seed for reproducibility in tests.
    """

    def __init__(
        self,
        random_rate: float = 0.01,
        principle_rate: float = 0.05,
        safety_rate: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not (0 <= random_rate <= 1):
            raise ValueError(f"random_rate must be in [0,1]; got {random_rate}")
        if not (0 <= principle_rate <= 1):
            raise ValueError(f"principle_rate must be in [0,1]; got {principle_rate}")
        if not (0 <= safety_rate <= 1):
            raise ValueError(f"safety_rate must be in [0,1]; got {safety_rate}")

        self.random_rate = random_rate
        self.principle_rate = principle_rate
        self.safety_rate = safety_rate
        self._rng = random.Random(seed)

    def should_sample(
        self, output: str, context: Dict[str, Any] | None = None
    ) -> Tuple[bool, str]:
        """Decide whether to sample *output* for alignment monitoring.

        The highest applicable sampling tier wins.

        Args:
            output: The raw model output string.
            context: Optional metadata dict.  A key ``"safety_flagged": True``
                forces the safety-triggered tier regardless of text analysis.

        Returns:
            A ``(decision, reason)`` tuple where *decision* is True if the
            output should be sampled and *reason* is a string identifying
            which sampling tier was applied.
        """
        ctx = context or {}
        output_lower = output.lower()

        # Safety-triggered: check explicit flag or keyword heuristic
        if ctx.get("safety_flagged") or any(k in output_lower for k in _SAFETY_KEYWORDS):
            sampled = self._rng.random() < self.safety_rate
            reason = "safety_triggered"
            logger.debug("Safety-triggered sampling: sampled=%s", sampled)
            return sampled, reason

        # Principle-triggered
        if any(k in output_lower for k in _PRINCIPLE_KEYWORDS):
            sampled = self._rng.random() < self.principle_rate
            reason = "principle_triggered"
            logger.debug("Principle-triggered sampling: sampled=%s", sampled)
            return sampled, reason

        # Random baseline
        sampled = self._rng.random() < self.random_rate
        reason = "random"
        logger.debug("Random sampling: sampled=%s", sampled)
        return sampled, reason

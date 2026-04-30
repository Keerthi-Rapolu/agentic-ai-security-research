"""CUSUM change-point detection for alignment regression monitoring."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CUSUMResult:
    """Result of a single CUSUM update step.

    Attributes:
        statistic: Current CUSUM statistic S_t.
        mmd2_value: The MMD² value that was ingested in this step.
        is_alert: Whether S_t crossed the decision threshold h.
        severity: Alert severity: ``"P1"``, ``"P2"``, ``"P3"``, or ``None``.
        step: Cumulative step counter since last reset.
    """

    statistic: float
    mmd2_value: float
    is_alert: bool
    severity: str | None
    step: int


class CUSUMDetector:
    """CUSUM (Cumulative Sum) change-point detector for MMD² time series.

    Implements the one-sided CUSUM update rule:

        S_t = max(0, S_{t-1} + MMD²_t - μ₀ - k)

    An alert fires when S_t > h.  The detector resets S_t = 0 after each
    alert to allow detection of subsequent regime changes.

    The decision threshold h and slack k are calibrated from ``mu_null``,
    ``sigma_null``, and the target ARL via:
        k = slack_factor × σ₀
        h is set by the caller or left at config default

    Args:
        mu_null: Expected MMD² under the null (no regression) hypothesis.
        sigma_null: Standard deviation of MMD² under null.
        slack_factor: Multiplier for σ₀ to derive the slack parameter k.
        decision_threshold: CUSUM statistic threshold h that triggers an alert.
        theta_align: MMD² threshold for severity scoring.
        theta_max: MMD² ceiling for severity scoring.
        arl_target: Target average run length (informational; used for logging).
    """

    def __init__(
        self,
        mu_null: float = 0.01,
        sigma_null: float = 0.005,
        slack_factor: float = 0.5,
        decision_threshold: float = 5.0,
        theta_align: float = 0.05,
        theta_max: float = 0.30,
        arl_target: int = 500,
    ) -> None:
        self.mu_null = mu_null
        self.sigma_null = sigma_null
        self.slack = slack_factor * sigma_null
        self.h = decision_threshold
        self.theta_align = theta_align
        self.theta_max = theta_max
        self.arl_target = arl_target

        self._statistic: float = 0.0
        self._step: int = 0

        logger.info(
            "CUSUMDetector init: μ₀=%.4f σ₀=%.4f k=%.4f h=%.4f ARL=%d",
            mu_null, sigma_null, self.slack, self.h, arl_target,
        )

    def update(self, mmd2_value: float) -> CUSUMResult:
        """Ingest one MMD² observation and return the detection result.

        Args:
            mmd2_value: MMD² estimate for the current monitoring window.

        Returns:
            :class:`CUSUMResult` with the updated statistic and alert status.
        """
        self._step += 1
        self._statistic = max(0.0, self._statistic + mmd2_value - self.mu_null - self.slack)
        is_alert = self._statistic > self.h
        severity = self._severity(mmd2_value) if is_alert else None

        result = CUSUMResult(
            statistic=self._statistic,
            mmd2_value=mmd2_value,
            is_alert=is_alert,
            severity=severity,
            step=self._step,
        )

        if is_alert:
            logger.warning(
                "CUSUM ALERT at step %d: S=%.4f > h=%.4f, MMD²=%.4f, severity=%s",
                self._step, self._statistic, self.h, mmd2_value, severity,
            )
            self.reset()
        else:
            logger.debug(
                "CUSUM step %d: S=%.4f, MMD²=%.4f",
                self._step, self._statistic, mmd2_value,
            )

        return result

    def reset(self) -> None:
        """Reset the CUSUM statistic to zero after an alert."""
        logger.info("CUSUM reset at step %d", self._step)
        self._statistic = 0.0

    @property
    def statistic(self) -> float:
        """Current CUSUM statistic value."""
        return self._statistic

    def _severity(self, mmd2: float) -> str:
        """Map an MMD² value to a P1/P2/P3 alert severity string."""
        span = self.theta_max - self.theta_align
        if span <= 0:
            return "P1"
        score = (mmd2 - self.theta_align) / span
        if score > 0.75:
            return "P1"
        if score > 0.40:
            return "P2"
        return "P3"

"""Alignment regression monitor — orchestrates sampling, embedding, MMD, and CUSUM."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from baseline import ConstitutionalBaseline
from changepoint import CUSUMDetector, CUSUMResult
from config import MonitorConfig
from mmd import MMDComputer
from sampler import OutputSampler

logger = logging.getLogger(__name__)


@dataclass
class MonitorResult:
    """Result from processing a single production output.

    Attributes:
        sampled: Whether the output was included in the monitoring window.
        sample_reason: Which sampling tier was applied.
        mmd2: MMD² estimate for the current window (None if window not full).
        cusum_result: CUSUM update result (None if window not full).
        window_size: Number of outputs in the current monitoring window.
        regression_type: ``"alignment"``, ``"capability"``, ``"mixed"``, or
            ``None`` if no regression detected.
    """

    sampled: bool
    sample_reason: str
    mmd2: Optional[float] = None
    cusum_result: Optional[CUSUMResult] = None
    window_size: int = 0
    regression_type: Optional[str] = None


@dataclass
class MonitorStatus:
    """Snapshot of the monitor's current internal state.

    Attributes:
        total_ingested: Total outputs ingested since the monitor was started.
        total_sampled: Outputs that passed the sampling filter.
        window_size: Current monitoring window size.
        cusum_statistic: Current CUSUM statistic value.
        last_mmd2: Most recent MMD² estimate.
        alert_count: Number of CUSUM alerts fired.
    """

    total_ingested: int
    total_sampled: int
    window_size: int
    cusum_statistic: float
    last_mmd2: Optional[float]
    alert_count: int


class AlignmentRegressionMonitor:
    """End-to-end alignment regression monitor.

    Orchestrates the full pipeline:
        1. **OutputSampler** — decides whether each production output is monitored.
        2. **SentenceTransformer** — embeds sampled outputs.
        3. **MMDComputer** — estimates distributional distance from the baseline.
        4. **CUSUMDetector** — detects change-points in the MMD² time series.

    When the monitoring window fills (``window_size`` sampled outputs), an
    MMD² estimate is computed and fed to the CUSUM detector.  The window then
    slides forward by discarding the oldest half.

    Args:
        baseline: A fitted :class:`ConstitutionalBaseline` instance.
        config: :class:`MonitorConfig` with all thresholds and settings.
        window_size: Number of sampled outputs per MMD² computation window.
    """

    def __init__(
        self,
        baseline: ConstitutionalBaseline,
        config: Optional[MonitorConfig] = None,
        window_size: int = 100,
    ) -> None:
        self.baseline = baseline
        self.config = config or MonitorConfig()
        self.window_size = window_size

        self._sampler = OutputSampler(
            random_rate=self.config.random_sample_rate,
            principle_rate=self.config.principle_sample_rate,
            safety_rate=self.config.safety_sample_rate,
        )
        self._mmd = MMDComputer()
        self._cusum = CUSUMDetector(
            theta_align=self.config.theta_align,
            theta_max=self.config.theta_max,
            decision_threshold=self.config.cusum_decision_threshold,
        )
        self._embedder = None

        self._buffer: List[np.ndarray] = []
        self._total_ingested: int = 0
        self._total_sampled: int = 0
        self._alert_count: int = 0
        self._last_mmd2: Optional[float] = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def ingest_output(self, output: str, context: Optional[Dict[str, Any]] = None) -> MonitorResult:
        """Process one production output through the monitoring pipeline.

        Args:
            output: Raw model output string.
            context: Optional metadata (e.g. ``{"safety_flagged": True}``).

        Returns:
            :class:`MonitorResult` with sampling decision and, if the window
            is full, the MMD² estimate and CUSUM result.
        """
        self._total_ingested += 1
        sampled, reason = self._sampler.should_sample(output, context)

        if not sampled:
            return MonitorResult(sampled=False, sample_reason=reason)

        self._total_sampled += 1
        model = self._get_embedder()
        emb = model.encode([output], convert_to_numpy=True, show_progress_bar=False)[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        self._buffer.append(emb.astype(np.float32))

        result = MonitorResult(
            sampled=True,
            sample_reason=reason,
            window_size=len(self._buffer),
        )

        if len(self._buffer) >= self.window_size:
            mmd2, cusum_result = self._evaluate_window()
            regression_type = self._classify_regression(mmd2, cusum_result)
            result.mmd2 = mmd2
            result.cusum_result = cusum_result
            result.regression_type = regression_type
            if cusum_result.is_alert:
                self._alert_count += 1

        return result

    def _evaluate_window(self):
        """Compute MMD² for the current buffer and update CUSUM."""
        production_emb = np.array(self._buffer, dtype=np.float32)
        baseline_emb = self.baseline.get_embeddings()

        mmd2 = self._mmd.compute(production_emb, baseline_emb)
        self._last_mmd2 = mmd2

        cusum_result = self._cusum.update(max(0.0, mmd2))

        # Slide window: drop oldest half
        drop = self.window_size // 2
        self._buffer = self._buffer[drop:]

        logger.info(
            "Window evaluated: MMD²=%.6f, CUSUM_S=%.4f, alert=%s",
            mmd2, cusum_result.statistic, cusum_result.is_alert,
        )
        return mmd2, cusum_result

    @staticmethod
    def _classify_regression(mmd2: Optional[float], cusum: Optional[CUSUMResult]) -> Optional[str]:
        """Heuristically classify the type of regression if an alert was raised.

        A full decomposition would require separate MMD computation on
        capability-only and alignment-only prompt subsets; here we use the
        MMD² magnitude as a proxy for now.
        """
        if cusum is None or not cusum.is_alert:
            return None
        if mmd2 is None:
            return None
        # Severity-based heuristic:
        # P1 likely combined, P2 likely alignment, P3 could be capability
        if cusum.severity == "P1":
            return "mixed"
        if cusum.severity == "P2":
            return "alignment"
        return "capability"

    def get_status(self) -> MonitorStatus:
        """Return a snapshot of the monitor's current state.

        Returns:
            :class:`MonitorStatus` data object.
        """
        return MonitorStatus(
            total_ingested=self._total_ingested,
            total_sampled=self._total_sampled,
            window_size=len(self._buffer),
            cusum_statistic=self._cusum.statistic,
            last_mmd2=self._last_mmd2,
            alert_count=self._alert_count,
        )

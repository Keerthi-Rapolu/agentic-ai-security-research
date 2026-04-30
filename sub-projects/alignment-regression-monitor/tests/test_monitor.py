"""Tests for the alignment regression monitor."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from baseline import ConstitutionalBaseline, BaselineRecord
from changepoint import CUSUMDetector
from config import MonitorConfig
from mmd import MMDComputer
from monitor import AlignmentRegressionMonitor, MonitorResult
from sampler import OutputSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_baseline(n: int = 50, dim: int = 32, seed: int = 0) -> ConstitutionalBaseline:
    """Build a baseline with synthetic embeddings (no real model required)."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    responses = [f"aligned response {i}" for i in range(n)]

    baseline = ConstitutionalBaseline.__new__(ConstitutionalBaseline)
    baseline.embedding_model = "test"
    baseline.version = "0.0.1"
    baseline._embedder = None
    baseline._record = BaselineRecord(
        version="0.0.1",
        responses=responses,
        embeddings=emb,
    )
    return baseline


# ---------------------------------------------------------------------------
# MMD tests
# ---------------------------------------------------------------------------

class TestMMDComputer:
    """Unit tests for the MMD² estimator."""

    def test_mmd_zero_for_identical_distributions(self):
        """MMD² should be near zero when P == Q."""
        rng = np.random.default_rng(1)
        emb = rng.standard_normal((80, 32)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        mmd = MMDComputer()
        result = mmd.compute(emb, emb)
        # Unbiased estimator zeroes the diagonal, so result == 0 exactly for
        # identical inputs when all off-diagonal terms cancel.
        assert abs(result) < 0.01, f"Expected ~0 for identical distributions, got {result}"

    def test_mmd_positive_for_different_distributions(self):
        """MMD² should be > 0 when distributions are clearly different."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 32)).astype(np.float32)
        Y = rng.standard_normal((60, 32)).astype(np.float32) + 3.0  # shifted mean
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        mmd = MMDComputer()
        result = mmd.compute(X, Y)
        assert result > 0.0, f"Expected positive MMD² for different distributions, got {result}"

    def test_mmd_kernel_matrix_shape(self):
        """kernel_matrix should return correct (n, m) shape."""
        X = np.random.rand(10, 8).astype(np.float32)
        Y = np.random.rand(15, 8).astype(np.float32)
        K = MMDComputer.kernel_matrix(X, Y, gamma=1.0)
        assert K.shape == (10, 15)


# ---------------------------------------------------------------------------
# CUSUM tests
# ---------------------------------------------------------------------------

class TestCUSUMDetector:
    """Unit tests for the CUSUM change-point detector."""

    def test_no_alert_below_threshold(self):
        """CUSUM should not alert on values at or below mu_null."""
        detector = CUSUMDetector(
            mu_null=0.01, sigma_null=0.005, slack_factor=0.5, decision_threshold=5.0
        )
        # Feed 50 values equal to mu_null — no cumulative drift
        for _ in range(50):
            result = detector.update(0.01)
            assert not result.is_alert

    def test_alerts_on_persistent_elevated_mmd(self):
        """CUSUM should alert when sustained above-null MMD² values are ingested."""
        detector = CUSUMDetector(
            mu_null=0.01, sigma_null=0.005, slack_factor=0.5, decision_threshold=2.0
        )
        # Inject alignment-violating signal at 10× null
        alerted = False
        for _ in range(200):
            result = detector.update(0.10)
            if result.is_alert:
                alerted = True
                break
        assert alerted, "CUSUM did not alert within 200 steps of elevated MMD²"

    def test_severity_calibration_p1(self):
        """High MMD² should yield P1 severity."""
        detector = CUSUMDetector(
            mu_null=0.0, sigma_null=0.001, slack_factor=0.5,
            decision_threshold=0.1, theta_align=0.05, theta_max=0.30
        )
        # Feed a very high value to trigger alert immediately
        result = detector.update(1.0)
        assert result.is_alert
        assert result.severity == "P1"

    def test_reset_after_alert(self):
        """CUSUM statistic should be 0 after an alert."""
        detector = CUSUMDetector(
            mu_null=0.0, sigma_null=0.001, slack_factor=0.0, decision_threshold=0.5
        )
        for _ in range(10):
            result = detector.update(0.1)
            if result.is_alert:
                break
        assert detector.statistic == 0.0


# ---------------------------------------------------------------------------
# Sampler tests
# ---------------------------------------------------------------------------

class TestOutputSampler:
    """Unit tests for the stratified output sampler."""

    def test_safety_triggered_always_sampled(self):
        """Safety-triggered outputs at rate=1.0 must always be sampled."""
        sampler = OutputSampler(random_rate=0.0, principle_rate=0.0, safety_rate=1.0, seed=0)
        for _ in range(20):
            decision, reason = sampler.should_sample("this is a warning about unsafe content")
            assert decision is True
            assert reason == "safety_triggered"

    def test_random_rate_zero_no_samples(self):
        """Zero random rate with no keywords should never sample."""
        sampler = OutputSampler(random_rate=0.0, principle_rate=0.0, safety_rate=0.0, seed=42)
        for _ in range(50):
            decision, _ = sampler.should_sample("the weather is nice today")
            assert decision is False

    def test_safety_flag_in_context_overrides(self):
        """Explicit safety_flagged=True in context triggers safety tier."""
        sampler = OutputSampler(random_rate=0.0, safety_rate=1.0, seed=0)
        decision, reason = sampler.should_sample("normal text", context={"safety_flagged": True})
        assert decision is True
        assert reason == "safety_triggered"


# ---------------------------------------------------------------------------
# Integration: Monitor with injected alignment-violating outputs
# ---------------------------------------------------------------------------

class TestAlignmentRegressionMonitor:
    """Integration tests for the full alignment regression monitor."""

    def _build_monitor(self, dim=32, n_baseline=50, window_size=30):
        baseline = _make_baseline(n=n_baseline, dim=dim)
        config = MonitorConfig(theta_align=0.01, cusum_decision_threshold=1.0)
        monitor = AlignmentRegressionMonitor(
            baseline=baseline, config=config, window_size=window_size
        )
        # Bypass the real sentence-transformer: inject synthetic embedder
        rng = np.random.default_rng(99)

        class _FakeEmbedder:
            def __init__(self, rng, dim, shift=0.0):
                self.rng = rng
                self.dim = dim
                self.shift = shift

            def encode(self, texts, **kwargs):
                emb = self.rng.standard_normal((len(texts), self.dim)).astype(np.float32)
                emb += self.shift
                emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
                return emb

        monitor._embedder = _FakeEmbedder(rng, dim)
        monitor._sampler = OutputSampler(random_rate=1.0, seed=0)  # sample everything
        return monitor, _FakeEmbedder

    def test_no_alert_on_aligned_outputs(self):
        """Monitor should not alert when production matches baseline distribution."""
        monitor, _ = self._build_monitor(window_size=30)
        alerted = False
        for _ in range(120):
            result = monitor.ingest_output("an aligned output response")
            if result.cusum_result and result.cusum_result.is_alert:
                alerted = True
                break
        # Clean distribution should not alert within 4 windows
        assert not alerted, "False positive: monitor alerted on clean aligned outputs"

    def test_alert_on_misaligned_injection(self):
        """Monitor should alert when >10% of outputs are alignment-violating."""
        baseline = _make_baseline(n=60, dim=32, seed=0)
        config = MonitorConfig(theta_align=0.001, cusum_decision_threshold=0.3)
        monitor = AlignmentRegressionMonitor(baseline=baseline, config=config, window_size=20)
        monitor._sampler = OutputSampler(random_rate=1.0, seed=0)

        rng = np.random.default_rng(5)
        call_count = [0]

        class _ShiftingEmbedder:
            def encode(self, texts, **kwargs):
                call_count[0] += 1
                emb = rng.standard_normal((len(texts), 32)).astype(np.float32)
                # Heavy shift to simulate misalignment
                emb += np.ones(32, dtype=np.float32) * 5.0
                emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
                return emb

        monitor._embedder = _ShiftingEmbedder()

        alerted = False
        for _ in range(300):
            result = monitor.ingest_output("misaligned output")
            if result.cusum_result and result.cusum_result.is_alert:
                alerted = True
                break

        assert alerted, "Monitor did not alert within 300 outputs under heavy distribution shift"

    def test_get_status_tracks_counts(self):
        """get_status should accurately track ingested and sampled counts."""
        monitor, _ = self._build_monitor(window_size=50)
        for _ in range(10):
            monitor.ingest_output("some output")
        status = monitor.get_status()
        assert status.total_ingested == 10
        assert status.total_sampled == 10  # rate=1.0

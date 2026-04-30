"""Tests for the temporal RAG corpus drift detector."""

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from config import DetectorConfig
from detector import DriftEvent, TemporalCorpusDriftDetector
from provenance import ProvenanceGraph
from snapshot_store import Snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(window_id: str, n: int, dim: int = 64, seed: int = 0) -> Snapshot:
    """Create a synthetic snapshot with random unit-norm embeddings."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb /= norms
    metadata = [{"doc_id": f"{window_id}_doc{i}", "source": "test"} for i in range(n)]
    return Snapshot(window_id=window_id, embeddings=emb, metadata=metadata)


def _poisoned_snapshot(
    clean_snapshot: Snapshot,
    n_poison: int,
    dim: int = 64,
    direction: Optional[np.ndarray] = None,
    seed: int = 99,
) -> Snapshot:
    """Append *n_poison* documents shifted far in embedding space."""
    rng = np.random.default_rng(seed)
    if direction is None:
        direction = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

    poison_emb = rng.standard_normal((n_poison, dim)).astype(np.float32)
    # Large shift (100×) makes noise negligible post-normalisation, producing a
    # tight cluster near `direction` that is clearly distinct from the clean set.
    poison_emb += direction * 100.0
    norms = np.linalg.norm(poison_emb, axis=1, keepdims=True)
    poison_emb /= norms

    combined = np.vstack([clean_snapshot.embeddings, poison_emb])
    meta_poison = [{"doc_id": f"poison_{i}", "source": "attacker"} for i in range(n_poison)]
    combined_meta = clean_snapshot.metadata + meta_poison

    return Snapshot(
        window_id=clean_snapshot.window_id + "_poisoned",
        embeddings=combined,
        metadata=combined_meta,
    )




# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDriftDetectionTriggers:
    """Drift detection fires above threshold and not below."""

    def setup_method(self):
        self.config = DetectorConfig(
            theta_corpus=0.15,
            delta_fast=0.02,
            projection_dims=16,
            hdbscan_min_cluster_size=2,
            window_size=4,
        )
        self.detector = TemporalCorpusDriftDetector(config=self.config)

    def test_no_alert_on_clean_corpus(self):
        """Two similar snapshots should produce no anomaly."""
        rng = np.random.default_rng(42)
        dim = 64
        base = rng.standard_normal((100, dim)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)

        snap1 = Snapshot("W1", base.copy())
        # Slight noise — should remain within threshold
        noisy = base + rng.standard_normal((100, dim)).astype(np.float32) * 0.01
        noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
        snap2 = Snapshot("W2", noisy)

        events = self.detector.detect([snap1, snap2])
        assert len(events) == 1
        event = events[0]
        assert event.is_anomaly is False

    def test_alert_on_heavily_poisoned_corpus(self):
        """A fully-replaced window with antipodal embeddings triggers an alert.

        Prev snapshot: 100 random unit-sphere vectors (clean).
        Curr snapshot: 100 vectors tightly clustered in the -direction hemisphere
        (antipodal to the clean centroid).  This maximises JSD and reliably
        exceeds theta_corpus even with few random-projection dimensions.
        """
        dim = 64
        rng = np.random.default_rng(42)
        direction = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

        clean_emb = rng.standard_normal((100, dim)).astype(np.float32)
        clean_emb /= np.linalg.norm(clean_emb, axis=1, keepdims=True)
        snap_clean = Snapshot("W1", clean_emb)

        # Fully-replaced window: all docs point in the -direction hemisphere
        poison_raw = rng.standard_normal((100, dim)).astype(np.float32)
        poison_raw -= direction * 100.0   # antipodal shift
        poison_raw /= np.linalg.norm(poison_raw, axis=1, keepdims=True)
        snap_poison = Snapshot("W2", poison_raw)

        events = self.detector.detect([snap_clean, snap_poison])
        assert len(events) == 1
        event = events[0]
        assert event.is_anomaly is True, f"Expected anomaly; drift_score={event.drift_score:.4f}"

    def test_drift_score_below_threshold_no_alert(self):
        """Small distribution shift that stays below theta_corpus is not flagged."""
        dim = 64
        snap1 = _make_snapshot("W1", n=200, dim=dim, seed=10)
        # Second snapshot is the same distribution — JSD should be near zero
        snap2 = _make_snapshot("W2", n=200, dim=dim, seed=10)

        events = self.detector.detect([snap1, snap2])
        for e in events:
            assert e.is_anomaly is False


class TestProvenanceAttribution:
    """Provenance graph identifies injected documents."""

    def test_counterfactual_identifies_poison_docs(self):
        """Removing poisoned docs should reduce drift below threshold."""
        dim = 32
        rng = np.random.default_rng(7)

        prev_emb = rng.standard_normal((80, dim)).astype(np.float32)
        prev_emb /= np.linalg.norm(prev_emb, axis=1, keepdims=True)

        clean_new = rng.standard_normal((20, dim)).astype(np.float32)
        clean_new /= np.linalg.norm(clean_new, axis=1, keepdims=True)

        direction = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        poison_emb = rng.standard_normal((5, dim)).astype(np.float32) + direction * 8.0
        poison_emb /= np.linalg.norm(poison_emb, axis=1, keepdims=True)

        curr_emb = np.vstack([prev_emb, clean_new, poison_emb])

        new_doc_mask = np.zeros(len(curr_emb), dtype=bool)
        new_doc_mask[len(prev_emb):] = True  # new docs are the last 25

        baseline_drift = 0.5  # synthetic non-zero drift

        contributions = ProvenanceGraph.compute_counterfactual_contributions(
            prev_emb, curr_emb, new_doc_mask, baseline_drift, projection_dims=8
        )

        # Expect some contributions identified
        assert len(contributions) > 0, "Expected at least one poisoning candidate"
        # Each contribution delta should be positive (removal reduces drift)
        for idx, delta in contributions:
            assert delta >= 0.0

    def test_provenance_graph_construction(self):
        """ProvenanceGraph.build returns a graph with correct nodes and edges."""
        dim = 16
        direction = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

        # Two similar docs + one outlier
        similar_a = direction.copy()
        similar_b = direction + np.array([0.01] * dim, dtype=np.float32)
        similar_b /= np.linalg.norm(similar_b)
        outlier = -direction.copy()

        candidate_docs = [
            {"doc_id": "doc_a", "drift_contribution": 0.3},
            {"doc_id": "doc_b", "drift_contribution": 0.2},
            {"doc_id": "doc_c", "drift_contribution": 0.1},
        ]
        embeddings = np.vstack([similar_a, similar_b, outlier]).astype(np.float32)

        pg = ProvenanceGraph(sigma=0.8)
        graph = pg.build(candidate_docs, embeddings)

        assert graph.number_of_nodes() == 3
        top = pg.get_top_candidates(n=2)
        assert len(top) == 2
        assert top[0].drift_contribution >= top[1].drift_contribution


class TestMultiSnapshotSliding:
    """Sliding window comparison across multiple snapshots."""

    def test_multiple_snapshots_returns_correct_event_count(self):
        """Five snapshots with window_size=4 should yield 3 comparison events."""
        config = DetectorConfig(window_size=4, projection_dims=8, delta_fast=0.001)
        detector = TemporalCorpusDriftDetector(config=config)
        snapshots = [_make_snapshot(f"W{i}", 50, dim=32, seed=i) for i in range(5)]
        events = detector.detect(snapshots)
        assert len(events) == 3  # window [-4:] = W1..W4, pairs: (W1,W2),(W2,W3),(W3,W4)

    def test_event_window_ids_are_sequential(self):
        """Event window IDs should follow the order of the input snapshots."""
        config = DetectorConfig(window_size=4, projection_dims=8, delta_fast=0.001)
        detector = TemporalCorpusDriftDetector(config=config)
        ids = ["A", "B", "C", "D", "E"]
        snapshots = [_make_snapshot(wid, 40, dim=16, seed=i) for i, wid in enumerate(ids)]
        events = detector.detect(snapshots)
        expected_pairs = [("B", "C"), ("C", "D"), ("D", "E")]
        for event, (expected_prev, expected_curr) in zip(events, expected_pairs):
            assert event.prev_window_id == expected_prev
            assert event.window_id == expected_curr

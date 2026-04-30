"""Temporal corpus drift detection for RAG corpus poisoning."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config import DetectorConfig
from provenance import ProvenanceGraph, _jsd_via_projection

logger = logging.getLogger(__name__)


@dataclass
class DriftEvent:
    """A detected corpus drift event.

    Attributes:
        window_id: The current snapshot window that triggered the alert.
        prev_window_id: The prior snapshot window used for comparison.
        drift_score: Observed JSD drift score.
        threshold: Configured detection threshold (theta_corpus).
        is_anomaly: Whether drift_score exceeds threshold.
        severity: Alert severity (``"P1"``, ``"P2"``, ``"P3"``, or ``"INFO"``).
        centroid_distance: L2 distance between snapshot centroids (Stage 1).
        poisoning_candidates: Docs identified as likely poisoning sources.
        provenance_graph: NetworkX DiGraph (if provenance was computed).
    """

    window_id: str
    prev_window_id: str
    drift_score: float
    threshold: float
    is_anomaly: bool
    severity: str = "INFO"
    centroid_distance: float = 0.0
    poisoning_candidates: List[Dict[str, Any]] = field(default_factory=list)
    provenance_graph: Any = None  # nx.DiGraph when computed


class TemporalCorpusDriftDetector:
    """Two-stage temporal drift detector for RAG corpus poisoning.

    Stage 1 performs a fast L2 centroid distance check; only snapshots that
    exceed ``delta_fast`` proceed to Stage 2.  Stage 2 computes Jensen-Shannon
    divergence between random-projected kernel density estimates of consecutive
    snapshot pairs.  When JSD exceeds ``theta_corpus``, HDBSCAN outlier
    detection and counterfactual provenance attribution are run.

    Args:
        config: :class:`DetectorConfig` with all thresholds and settings.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._provenance = ProvenanceGraph(
            sigma=self.config.sigma_similarity,
            max_candidates=self.config.max_candidate_docs,
        )

    def detect(self, snapshots: list) -> List[DriftEvent]:
        """Run drift detection over a chronological list of snapshots.

        Each snapshot must have attributes (or dict keys):
        ``window_id`` (str) and ``embeddings`` (np.ndarray, shape N×D).
        Optional ``metadata`` (list of dicts) provides per-document provenance.

        A sliding window of ``config.window_size`` is applied: each snapshot
        is compared to the immediately preceding one within the window.

        Args:
            snapshots: Ordered list of snapshot objects or dicts.

        Returns:
            List of :class:`DriftEvent` instances for each comparison pair,
            including non-anomalous pairs (``is_anomaly=False``).
        """
        events: List[DriftEvent] = []

        if len(snapshots) < 2:
            logger.info("Need at least 2 snapshots for drift detection; got %d.", len(snapshots))
            return events

        # Respect sliding window size
        window = snapshots[-self.config.window_size :]

        for i in range(1, len(window)):
            prev = window[i - 1]
            curr = window[i]
            event = self._compare_pair(prev, curr)
            events.append(event)

        return events

    def _compare_pair(self, prev, curr) -> DriftEvent:
        """Compare one snapshot pair and return a :class:`DriftEvent`."""
        prev_id = _attr(prev, "window_id")
        curr_id = _attr(curr, "window_id")
        prev_emb: np.ndarray = _attr(prev, "embeddings")
        curr_emb: np.ndarray = _attr(curr, "embeddings")

        if len(prev_emb) == 0 or len(curr_emb) == 0:
            logger.warning("Empty embeddings in window pair %s → %s", prev_id, curr_id)
            return DriftEvent(
                window_id=curr_id,
                prev_window_id=prev_id,
                drift_score=0.0,
                threshold=self.config.theta_corpus,
                is_anomaly=False,
            )

        # Stage 1 — fast centroid check
        mu_prev = prev_emb.mean(axis=0)
        mu_curr = curr_emb.mean(axis=0)
        centroid_dist = float(np.linalg.norm(mu_curr - mu_prev))

        if centroid_dist < self.config.delta_fast:
            logger.debug(
                "Pair %s→%s: centroid dist %.4f < delta_fast %.4f — skipped.",
                prev_id, curr_id, centroid_dist, self.config.delta_fast,
            )
            return DriftEvent(
                window_id=curr_id,
                prev_window_id=prev_id,
                drift_score=0.0,
                threshold=self.config.theta_corpus,
                is_anomaly=False,
                centroid_distance=centroid_dist,
            )

        # Stage 2 — JSD via random projection
        drift_score = self._compute_jsd(prev_emb, curr_emb)
        is_anomaly = drift_score > self.config.theta_corpus
        severity = self._severity(drift_score)

        logger.info(
            "Pair %s→%s: centroid_dist=%.4f JSD=%.4f threshold=%.4f anomaly=%s severity=%s",
            prev_id, curr_id, centroid_dist, drift_score,
            self.config.theta_corpus, is_anomaly, severity,
        )

        candidates: List[Dict[str, Any]] = []
        graph = None

        if is_anomaly:
            candidates, graph = self._run_provenance(prev_emb, curr_emb, curr, drift_score)

        return DriftEvent(
            window_id=curr_id,
            prev_window_id=prev_id,
            drift_score=drift_score,
            threshold=self.config.theta_corpus,
            is_anomaly=is_anomaly,
            severity=severity,
            centroid_distance=centroid_dist,
            poisoning_candidates=candidates,
            provenance_graph=graph,
        )

    def _compute_jsd(self, prev_emb: np.ndarray, curr_emb: np.ndarray) -> float:
        """Compute JSD between two embedding matrices via random projection."""
        from sklearn.random_projection import GaussianRandomProjection

        all_emb = np.vstack([prev_emb, curr_emb])
        n_features = all_emb.shape[1]
        target_dim = min(self.config.projection_dims, n_features)

        rp = GaussianRandomProjection(n_components=target_dim, random_state=42)
        rp.fit(all_emb)
        return _jsd_via_projection(prev_emb, curr_emb, rp)

    def _run_provenance(
        self,
        prev_emb: np.ndarray,
        curr_emb: np.ndarray,
        curr_snapshot,
        baseline_drift: float,
    ):
        """Identify poisoning candidates and build the provenance graph."""
        curr_metadata: List[Dict[str, Any]] = _attr(curr_snapshot, "metadata", default=[])
        n_prev = len(prev_emb)
        n_curr = len(curr_emb)

        # Documents new to the current snapshot
        new_doc_mask = np.zeros(n_curr, dtype=bool)
        if n_curr > n_prev:
            new_doc_mask[n_prev:] = True
        else:
            new_doc_mask[:] = True  # treat all as new when sizes unknown

        # HDBSCAN outlier detection on new documents
        new_indices = np.where(new_doc_mask)[0]
        hdbscan_noise: np.ndarray = np.array([], dtype=int)

        if len(new_indices) >= self.config.hdbscan_min_cluster_size:
            new_embeddings = curr_emb[new_indices]
            hdbscan_noise = self._hdbscan_outliers(new_embeddings, new_indices)

        # Counterfactual attribution
        from provenance import ProvenanceGraph

        pg = ProvenanceGraph(sigma=self.config.sigma_similarity)
        contributions = pg.compute_counterfactual_contributions(
            prev_emb,
            curr_emb,
            new_doc_mask,
            baseline_drift,
            self.config.projection_dims,
        )

        # Merge HDBSCAN noise indices into candidates
        candidate_indices = set(idx for idx, _ in contributions)
        for idx in hdbscan_noise:
            candidate_indices.add(int(idx))

        candidate_docs = []
        candidate_embeddings_list = []
        for idx in sorted(candidate_indices):
            meta = curr_metadata[idx] if idx < len(curr_metadata) else {}
            contrib = next((c for i, c in contributions if i == idx), 0.0)
            candidate_docs.append(
                {"doc_id": meta.get("doc_id", f"doc_{idx}"), "drift_contribution": contrib, **meta}
            )
            candidate_embeddings_list.append(curr_emb[idx])

        if not candidate_docs:
            return [], None

        cand_emb = np.array(candidate_embeddings_list, dtype=np.float32)
        graph = pg.build(candidate_docs, cand_emb)
        return candidate_docs, graph

    def _hdbscan_outliers(
        self, embeddings: np.ndarray, original_indices: np.ndarray
    ) -> np.ndarray:
        """Return indices (in original_indices) assigned to HDBSCAN noise class."""
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(embeddings)
            noise_mask = labels == -1
            return original_indices[noise_mask]
        except Exception as exc:
            logger.warning("HDBSCAN failed (%s); skipping outlier detection.", exc)
            return np.array([], dtype=int)

    def _severity(self, drift_score: float) -> str:
        """Map a JSD drift score to an alert severity level."""
        if drift_score >= self.config.severity_p1_threshold:
            return "P1"
        if drift_score >= self.config.severity_p2_threshold:
            return "P2"
        if drift_score >= self.config.theta_corpus:
            return "P3"
        return "INFO"


def _attr(obj, name: str, default=None):
    """Retrieve an attribute by name from an object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

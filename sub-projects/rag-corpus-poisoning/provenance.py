"""Provenance graph construction for RAG corpus poisoning attribution."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceNode:
    """A document node in the provenance graph.

    Attributes:
        doc_id: Unique document identifier.
        drift_contribution: How much drift this doc contributes (counterfactual delta).
        metadata: Provenance metadata (source, ingest time, etc.).
        is_candidate: Whether this doc is identified as a poisoning candidate.
    """

    doc_id: str
    drift_contribution: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_candidate: bool = False


class ProvenanceGraph:
    """Constructs and queries a document provenance graph for drift attribution.

    For each detected drift event, this class:
    1. Identifies new documents (those in the current snapshot but not the prior).
    2. Computes the counterfactual drift reduction for each new document.
    3. Builds a NetworkX DiGraph where nodes are documents and edges connect
       semantically similar docs above the ``sigma`` similarity threshold.

    Args:
        sigma: Cosine similarity threshold for drawing provenance graph edges.
        max_candidates: Maximum number of documents to evaluate counterfactually.
    """

    def __init__(self, sigma: float = 0.85, max_candidates: int = 50) -> None:
        self.sigma = sigma
        self.max_candidates = max_candidates
        self._graph = None  # lazy import networkx

    def build(
        self,
        candidate_docs: List[Dict[str, Any]],
        candidate_embeddings: np.ndarray,
        sigma: Optional[float] = None,
    ):
        """Build a provenance graph from candidate poisoning documents.

        Each document becomes a node. Edges are drawn between pairs whose
        cosine similarity exceeds *sigma*, indicating possible coordinated
        injection.

        Args:
            candidate_docs: List of dicts with at minimum a ``doc_id`` key and
                optional provenance metadata (``source``, ``ingest_ts``, etc.).
            candidate_embeddings: Float32 array of shape
                ``(len(candidate_docs), embedding_dim)`` containing the L2-
                normalised embeddings for each candidate.
            sigma: Cosine similarity threshold override. Defaults to
                ``self.sigma``.

        Returns:
            A ``networkx.DiGraph`` whose nodes carry :class:`ProvenanceNode`
            attributes and edges carry a ``similarity`` weight.
        """
        import networkx as nx

        threshold = sigma if sigma is not None else self.sigma
        graph = nx.DiGraph()

        for idx, doc in enumerate(candidate_docs):
            doc_id = doc.get("doc_id", str(idx))
            node = ProvenanceNode(
                doc_id=doc_id,
                drift_contribution=doc.get("drift_contribution", 0.0),
                metadata={k: v for k, v in doc.items() if k != "doc_id"},
                is_candidate=doc.get("drift_contribution", 0.0) > 0.0,
            )
            graph.add_node(doc_id, data=node)

        # Draw similarity edges between candidates
        n = len(candidate_docs)
        if n > 1 and len(candidate_embeddings):
            # cosine similarity matrix (embeddings are already L2-normalised)
            sim_matrix = candidate_embeddings @ candidate_embeddings.T
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(sim_matrix[i, j])
                    if sim >= threshold:
                        id_i = candidate_docs[i].get("doc_id", str(i))
                        id_j = candidate_docs[j].get("doc_id", str(j))
                        graph.add_edge(id_i, id_j, similarity=sim)
                        graph.add_edge(id_j, id_i, similarity=sim)

        self._graph = graph
        logger.debug(
            "Built provenance graph: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    def get_top_candidates(self, n: int = 10) -> List[ProvenanceNode]:
        """Return the top-n documents sorted by drift contribution (descending).

        Args:
            n: Number of top candidates to return.

        Returns:
            List of :class:`ProvenanceNode` instances sorted by
            ``drift_contribution`` descending.

        Raises:
            RuntimeError: If :meth:`build` has not been called first.
        """
        if self._graph is None:
            raise RuntimeError("Call build() before get_top_candidates().")

        nodes = [
            data["data"]
            for _, data in self._graph.nodes(data=True)
            if "data" in data
        ]
        nodes.sort(key=lambda node: node.drift_contribution, reverse=True)
        return nodes[:n]

    def get_connected_clusters(self) -> List[List[str]]:
        """Return weakly connected components as lists of doc IDs.

        Returns:
            List of clusters; each cluster is a list of doc IDs that are
            connected by similarity edges, suggesting coordinated injection.
        """
        if self._graph is None:
            return []

        import networkx as nx

        return [
            list(component)
            for component in nx.weakly_connected_components(self._graph)
            if len(component) > 1
        ]

    @staticmethod
    def compute_counterfactual_contributions(
        prev_embeddings: np.ndarray,
        curr_embeddings: np.ndarray,
        new_doc_mask: np.ndarray,
        baseline_drift: float,
        projection_dims: int = 32,
    ) -> List[Tuple[int, float]]:
        """Compute per-document drift contribution via counterfactual removal.

        For each new document (indicated by *new_doc_mask*), compute the JSD
        drift that would remain if that document were removed from the current
        snapshot.  Documents whose removal drives JSD below the baseline
        contribute positively to drift.

        Args:
            prev_embeddings: Embeddings from the prior snapshot.
            curr_embeddings: Embeddings from the current snapshot (including
                new documents).
            new_doc_mask: Boolean array of shape ``(len(curr_embeddings),)``
                where ``True`` marks a document new to the current snapshot.
            baseline_drift: The observed JSD drift before any removal.
            projection_dims: Random projection dimensions for JSD estimation.

        Returns:
            List of ``(doc_index, contribution_delta)`` tuples for documents
            that reduce drift when removed (positive contribution).
        """
        from sklearn.random_projection import GaussianRandomProjection

        new_indices = np.where(new_doc_mask)[0]
        if len(new_indices) == 0:
            return []

        rp = GaussianRandomProjection(n_components=projection_dims, random_state=42)
        all_emb = np.vstack([prev_embeddings, curr_embeddings])
        rp.fit(all_emb)

        contributions: List[Tuple[int, float]] = []
        for idx in new_indices:
            reduced = np.delete(curr_embeddings, idx, axis=0)
            if len(reduced) == 0:
                continue
            cf_drift = _jsd_via_projection(prev_embeddings, reduced, rp)
            delta = baseline_drift - cf_drift
            if delta > 0:
                contributions.append((int(idx), float(delta)))

        contributions.sort(key=lambda t: t[1], reverse=True)
        return contributions


def _jsd_via_projection(
    X: np.ndarray, Y: np.ndarray, rp
) -> float:
    """Estimate Jensen-Shannon divergence between two embedding sets.

    Projects both sets into a lower-dimensional space using the supplied
    random projection, then estimates JSD via histogram-based KDE on each
    projected dimension.

    Args:
        X: First embedding matrix.
        Y: Second embedding matrix.
        rp: Fitted ``GaussianRandomProjection`` instance.

    Returns:
        Scalar JSD estimate in ``[0, 1]``.
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import gaussian_kde

    Xp = rp.transform(X)
    Yp = rp.transform(Y)

    jsd_values = []
    n_dims = Xp.shape[1]
    grid_points = 50

    for d in range(n_dims):
        x_col = Xp[:, d]
        y_col = Yp[:, d]
        lo = min(x_col.min(), y_col.min())
        hi = max(x_col.max(), y_col.max())
        if hi == lo:
            continue
        grid = np.linspace(lo, hi, grid_points)
        try:
            px = gaussian_kde(x_col)(grid)
            py = gaussian_kde(y_col)(grid)
        except Exception:
            continue
        px = np.clip(px, 1e-10, None)
        py = np.clip(py, 1e-10, None)
        px /= px.sum()
        py /= py.sum()
        jsd_values.append(float(jensenshannon(px, py) ** 2))

    return float(np.mean(jsd_values)) if jsd_values else 0.0

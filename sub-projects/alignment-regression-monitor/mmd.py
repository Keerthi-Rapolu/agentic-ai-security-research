"""Maximum Mean Discrepancy (MMD²) estimator for alignment regression detection."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MMDComputer:
    """Unbiased U-statistic MMD² estimator with RBF kernel.

    Implements the unbiased estimator:

        MMD²(P, Q) = (1/s(s-1)) Σ_{i≠j} k(o_i, o_j)
                   - (2/sm)     Σ_{i,l} k(o_i, b_l)
                   + (1/m(m-1)) Σ_{l≠l'} k(b_l, b_{l'})

    where ``k(x, y) = exp(-γ ||x - y||²)`` and γ is chosen by the median
    heuristic on the pooled sample.

    Args:
        gamma: Fixed RBF bandwidth γ.  When ``None`` (default) the median
            heuristic is applied at each :meth:`compute` call.
    """

    def __init__(self, gamma: Optional[float] = None) -> None:
        self.gamma = gamma

    def compute(
        self,
        production_embeddings: np.ndarray,
        baseline_embeddings: np.ndarray,
    ) -> float:
        """Compute the unbiased MMD² estimate between production and baseline.

        Args:
            production_embeddings: Float32 array of shape ``(s, d)`` containing
                sampled production output embeddings.
            baseline_embeddings: Float32 array of shape ``(m, d)`` containing
                constitutional baseline embeddings.

        Returns:
            Scalar unbiased MMD² estimate.  May be slightly negative for very
            similar distributions; clip at zero downstream if needed.
        """
        if len(production_embeddings) < 2 or len(baseline_embeddings) < 2:
            logger.warning(
                "MMD requires ≥2 samples; got production=%d baseline=%d",
                len(production_embeddings),
                len(baseline_embeddings),
            )
            return 0.0

        gamma = self.gamma if self.gamma is not None else self._median_bandwidth(
            production_embeddings, baseline_embeddings
        )

        kpp = self.kernel_matrix(production_embeddings, production_embeddings, gamma)
        kpb = self.kernel_matrix(production_embeddings, baseline_embeddings, gamma)
        kbb = self.kernel_matrix(baseline_embeddings, baseline_embeddings, gamma)

        s = len(production_embeddings)
        m = len(baseline_embeddings)

        # Unbiased U-statistic — zero out diagonal
        np.fill_diagonal(kpp, 0.0)
        np.fill_diagonal(kbb, 0.0)

        term_pp = kpp.sum() / (s * (s - 1))
        term_pb = kpb.sum() / (s * m)
        term_bb = kbb.sum() / (m * (m - 1))

        mmd2 = term_pp - 2.0 * term_pb + term_bb
        logger.debug("MMD²=%.6f (γ=%.4f, s=%d, m=%d)", mmd2, gamma, s, m)
        return float(mmd2)

    @staticmethod
    def kernel_matrix(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute the RBF kernel matrix K(X, Y).

        K[i, j] = exp(-γ * ||X[i] - Y[j]||²)

        Args:
            X: Array of shape ``(n, d)``.
            Y: Array of shape ``(m, d)``.
            gamma: RBF bandwidth parameter γ.

        Returns:
            Float64 array of shape ``(n, m)``.
        """
        # ||x - y||² = ||x||² + ||y||² - 2 x·y  (broadcasting, numerically stable)
        xx = (X ** 2).sum(axis=1, keepdims=True)
        yy = (Y ** 2).sum(axis=1, keepdims=True)
        sq_dists = xx + yy.T - 2.0 * (X @ Y.T)
        sq_dists = np.clip(sq_dists, 0.0, None)  # numerical guard
        return np.exp(-gamma * sq_dists)

    @staticmethod
    def _median_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
        """Select RBF bandwidth γ via the median heuristic on the pooled sample.

        γ = 1 / (2 * median(||x - y||²))

        Args:
            X: Array of shape ``(n, d)``.
            Y: Array of shape ``(m, d)``.

        Returns:
            Scalar γ value; falls back to 1.0 if median is zero.
        """
        # Sub-sample for speed when datasets are large
        max_n = 500
        if len(X) > max_n:
            X = X[np.random.choice(len(X), max_n, replace=False)]
        if len(Y) > max_n:
            Y = Y[np.random.choice(len(Y), max_n, replace=False)]

        pooled = np.vstack([X, Y])
        n = len(pooled)
        # Pairwise squared distances on a random subsample of pairs
        max_pairs = 10000
        idx1 = np.random.randint(0, n, min(max_pairs, n * (n - 1) // 2))
        idx2 = np.random.randint(0, n, min(max_pairs, n * (n - 1) // 2))
        sq_dists = np.sum((pooled[idx1] - pooled[idx2]) ** 2, axis=1)
        median_sq = float(np.median(sq_dists))
        if median_sq <= 0:
            return 1.0
        return float(1.0 / (2.0 * median_sq))

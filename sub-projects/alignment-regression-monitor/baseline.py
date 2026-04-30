"""Constitutional baseline management for the alignment regression monitor."""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BaselineRecord:
    """A versioned, signed constitutional baseline.

    Attributes:
        version: Semantic version string for this baseline snapshot.
        responses: Raw aligned response strings used to build the baseline.
        embeddings: Float32 array of shape ``(n_responses, embedding_dim)``.
        principles: List of constitutional principle category labels.
        sha256: SHA-256 hash of the serialised baseline content for integrity.
        metadata: Optional free-form metadata (author, date, model tag, etc.).
    """

    version: str
    responses: List[str]
    embeddings: np.ndarray
    principles: List[str] = field(default_factory=list)
    sha256: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 over version + responses + embedding bytes."""
        h = hashlib.sha256()
        h.update(self.version.encode())
        for r in self.responses:
            h.update(r.encode())
        h.update(self.embeddings.tobytes())
        return h.hexdigest()

    def verify(self) -> bool:
        """Return True if the stored SHA-256 matches the current content."""
        return self.sha256 == self._compute_hash()


class ConstitutionalBaseline:
    """Manages the constitutional baseline for alignment regression monitoring.

    The baseline is a curated set of aligned model responses embedded in
    semantic space.  It is versioned and signed with a SHA-256 hash so that
    any tampering is detectable.

    Args:
        embedding_model: Sentence-transformer model identifier.
        version: Semantic version for the baseline.
    """

    def __init__(
        self,
        embedding_model: str = "all-mpnet-base-v2",
        version: str = "1.0.0",
    ) -> None:
        self.embedding_model = embedding_model
        self.version = version
        self._record: Optional[BaselineRecord] = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def embed_baseline(self, responses: List[str]) -> np.ndarray:
        """Embed a list of aligned responses and return L2-normalised vectors.

        Args:
            responses: Constitutional baseline response strings.

        Returns:
            Float32 array of shape ``(len(responses), embedding_dim)``.
        """
        model = self._get_embedder()
        embeddings = model.encode(responses, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (embeddings / norms).astype(np.float32)

    def fit(
        self,
        responses: List[str],
        principles: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> BaselineRecord:
        """Build and store a new baseline from the provided responses.

        Args:
            responses: Curated, constitutionally aligned response strings.
            principles: Constitutional principle category for each response.
            metadata: Arbitrary provenance metadata.

        Returns:
            The constructed :class:`BaselineRecord`.
        """
        if not responses:
            raise ValueError("responses list must not be empty")

        embeddings = self.embed_baseline(responses)
        self._record = BaselineRecord(
            version=self.version,
            responses=responses,
            embeddings=embeddings,
            principles=principles or [],
            metadata=metadata or {},
        )
        logger.info(
            "Baseline v%s built: %d responses, sha256=%s",
            self.version,
            len(responses),
            self._record.sha256[:16] + "…",
        )
        return self._record

    def get_embeddings(self) -> np.ndarray:
        """Return the baseline embedding matrix.

        Returns:
            Float32 array of shape ``(n_responses, embedding_dim)``.

        Raises:
            RuntimeError: If :meth:`fit` or :meth:`load` has not been called.
        """
        if self._record is None:
            raise RuntimeError("Baseline not loaded. Call fit() or load() first.")
        return self._record.embeddings

    def save(self, path: str) -> None:
        """Serialise the baseline to a pickle file.

        Args:
            path: File path for the saved baseline.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._record is None:
            raise RuntimeError("No baseline to save. Call fit() first.")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            pickle.dump(self._record, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Baseline saved to %s", dest)

    def load(self, path: str) -> BaselineRecord:
        """Load a previously saved baseline from a pickle file.

        Args:
            path: File path of the saved baseline.

        Returns:
            The loaded :class:`BaselineRecord`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the SHA-256 integrity check fails.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Baseline file not found: {path}")
        with open(src, "rb") as fh:
            record: BaselineRecord = pickle.load(fh)
        if not record.verify():
            raise ValueError(f"Baseline integrity check failed for {path}. File may be tampered.")
        self._record = record
        logger.info(
            "Baseline v%s loaded from %s (%d responses)",
            record.version,
            src,
            len(record.responses),
        )
        return record

"""Document embedding generation for the RAG corpus poisoning detector."""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Generates L2-normalised sentence embeddings for RAG corpus documents.

    Attempts to load the instruction-tuned ``instructor-xl`` model first;
    falls back to ``all-mpnet-base-v2`` if it is unavailable (e.g. in CI or
    low-resource environments).

    Args:
        model_name: Primary sentence-transformer model identifier.
        fallback_model: Fallback model identifier used when primary fails.
        batch_size: Number of documents processed per forward pass.
    """

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-xl",
        fallback_model: str = "all-mpnet-base-v2",
        batch_size: int = 32,
    ) -> None:
        self.batch_size = batch_size
        self._load_model(model_name, fallback_model)

    def _load_model(self, model_name: str, fallback_model: str) -> None:
        """Load the sentence-transformer model, falling back on any failure."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info("Loaded embedding model: %s", model_name)
        except Exception as exc:
            logger.warning(
                "Failed to load %s (%s). Falling back to %s.",
                model_name,
                exc,
                fallback_model,
            )
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(fallback_model)
            self.model_name = fallback_model
            logger.info("Loaded fallback embedding model: %s", fallback_model)

    def embed(self, documents: List[str]) -> np.ndarray:
        """Embed a list of documents and return L2-normalised vectors.

        Args:
            documents: Raw text documents to embed.

        Returns:
            Float32 array of shape ``(len(documents), embedding_dim)`` with
            each row L2-normalised to unit length.
        """
        if not documents:
            return np.empty((0,), dtype=np.float32)

        embeddings: np.ndarray = self._model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return self._l2_normalize(embeddings)

    def embed_single(self, document: str) -> np.ndarray:
        """Embed a single document and return its L2-normalised vector.

        Args:
            document: Raw text document to embed.

        Returns:
            Float32 array of shape ``(embedding_dim,)``.
        """
        return self.embed([document])[0]

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalise each row of an embedding matrix in-place."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms = np.where(norms == 0.0, 1.0, norms)
        return (embeddings / norms).astype(np.float32)

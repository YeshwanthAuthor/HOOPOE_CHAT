from typing import List, Optional, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from modules.utils import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Loads the cross-encoder model once (in __init__) and reuses it for
    every call - this is why it's a class and not a plain function:
    constructing a CrossEncoder per query would reload the model from disk
    (or re-download it) every time.
    """

    def __init__(self, config: dict):
        reranking_config = config["reranking"]
        self.final_k = reranking_config["final_k"]
        self.max_chars = reranking_config["max_chunk_chars"]
        model_name = reranking_config["cross_encoder_model"]
        logger.info("Loading cross-encoder model: %s", model_name)
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            logger.error("Failed to load cross-encoder model %s: %s", model_name, e)
            raise RuntimeError(
                f"Failed to load cross-encoder model '{model_name}'. Check your network "
                f"connection (the model downloads on first use) and the "
                f"reranking.cross_encoder_model value in config.yaml. Original error: {e}"
            ) from e

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Score every (query, chunk) pair with the cross-encoder and return
        the top_k (or self.final_k) Documents, sorted by score descending,
        as (Document, score) tuples.
        """
        if not docs:
            return []

        k = top_k if top_k is not None else self.final_k
        pairs = [(query, doc.page_content.strip()[: self.max_chars]) for doc in docs]

        logger.info("Reranking %d candidates for query: %r", len(docs), query)
        scores = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)

        ranked = sorted(zip(docs, scores.tolist()), key=lambda pair: pair[1], reverse=True)
        return ranked[:k]

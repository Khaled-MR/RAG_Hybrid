"""
BGE-reranker-v2-m3 wrapper.

A cross-encoder reranker that scores (query, document) pairs jointly.
This is the single biggest quality improvement you can add to a vector-only
RAG system: take top-20 from retrieval, rerank, return top-5.

Defaults to CPU so it doesn't compete with the LLM for GPU memory.
"""

from typing import List, Tuple
from FlagEmbedding import FlagReranker


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = False,
        device: str = "cpu",
    ):
        self.model = FlagReranker(
            model_name,
            use_fp16=use_fp16,
            devices=[device],
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Score (query, doc) pairs and return the top_k results.

        Returns a list of (original_index_into_documents, score),
        sorted by score descending.
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(pairs, normalize=True)

        # compute_score returns a float for a single pair, list otherwise
        if isinstance(scores, (int, float)):
            scores = [float(scores)]

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

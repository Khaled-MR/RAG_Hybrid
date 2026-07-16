
from typing import Dict, List, Tuple
import numpy as np
from FlagEmbedding import BGEM3FlagModel

# A sparse vector as Qdrant wants it: parallel lists of token ids and weights.
SparseVec = Tuple[List[int], List[float]]


def _to_sparse(lexical_weights: Dict) -> SparseVec:
    """Convert BGE-M3 lexical_weights ({token_id: weight}) to (indices, values)."""
    indices, values = [], []
    for token_id, weight in lexical_weights.items():
        w = float(weight)
        if w <= 0:
            continue
        indices.append(int(token_id))
        values.append(w)
    return indices, values


class BGEEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str = "cuda",
    ):

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            devices=[device],
        )

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(result["dense_vecs"], dtype=np.float32)

    def embed_documents_hybrid(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> Tuple[np.ndarray, List[SparseVec]]:
        """Return both dense vectors and BGE-M3 sparse (lexical) vectors."""
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(result["dense_vecs"], dtype=np.float32)
        sparse = [_to_sparse(lw) for lw in result["lexical_weights"]]
        return dense, sparse

    def embed_query(self, text: str) -> np.ndarray:
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(result["dense_vecs"][0], dtype=np.float32)

    def embed_query_hybrid(self, text: str) -> Tuple[np.ndarray, SparseVec]:
        """Return both the dense and sparse (lexical) vectors for a query."""
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(result["dense_vecs"][0], dtype=np.float32)
        sparse = _to_sparse(result["lexical_weights"][0])
        return dense, sparse

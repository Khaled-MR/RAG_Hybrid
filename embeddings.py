
from typing import List
import numpy as np
from FlagEmbedding import BGEM3FlagModel


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

    def embed_query(self, text: str) -> np.ndarray:
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(result["dense_vecs"][0], dtype=np.float32)

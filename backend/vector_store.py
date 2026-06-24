"""
LanceDB store with hybrid (vector + full-text) search.

Vector search finds semantic matches; full-text search (BM25 via Tantivy)
catches exact keyword/acronym matches that embeddings often miss. We fuse
both ranked lists with Reciprocal Rank Fusion (RRF) — a simple, robust
algorithm that doesn't need score calibration between the two retrievers.
"""

from typing import List, Dict, Any
import numpy as np
import pyarrow as pa
import lancedb


class HybridStore:
    def __init__(self, db_path: str, table_name: str, embedding_dim: int):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.table = None

    def _schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("metadata", pa.string()),  # JSON-encoded
            pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
        ])

    def create_or_open(self):
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
        else:
            self.table = self.db.create_table(self.table_name, schema=self._schema())
        return self.table

    def add(self, records: List[Dict[str, Any]]) -> None:
        if self.table is None:
            self.create_or_open()
        self.table.add(records)

    def build_fts_index(self, field: str = "text") -> None:
        """Build the Tantivy-backed BM25 index. Call after all docs are added."""
        if self.table is None:
            self.create_or_open()
        self.table.create_fts_index(field, replace=True)

    # --- individual retrievers ---

    def vector_search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Dict]:
        return (
            self.table.search(query_vector, vector_column_name="vector")
            .limit(top_k)
            .to_list()
        )

    def fts_search(self, query_text: str, top_k: int = 20) -> List[Dict]:
        try:
            return (
                self.table.search(query_text, query_type="fts")
                .limit(top_k)
                .to_list()
            )
        except Exception:
            # FTS index might not exist yet, or the query has no matches.
            return []

    # --- hybrid (RRF fusion) ---

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 20,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion: score = sum(weight / (rrf_k + rank))
        across both retrievers. Rank is 1-indexed by convention.
        """
        vec_hits = self.vector_search(query_vector, top_k=top_k)
        fts_hits = self.fts_search(query_text, top_k=top_k)

        scores: Dict[str, float] = {}
        items: Dict[str, Dict] = {}

        for rank, item in enumerate(vec_hits, start=1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + vector_weight / (rrf_k + rank)
            items[doc_id] = item

        for rank, item in enumerate(fts_hits, start=1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight / (rrf_k + rank)
            items.setdefault(doc_id, item)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [
            {"rrf_score": scores[doc_id], **items[doc_id]}
            for doc_id in ranked_ids[:top_k]
        ]

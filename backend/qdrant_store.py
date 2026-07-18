"""
Qdrant store (embedded / local file mode) with hybrid search.

Dense vectors (BGE-M3) capture meaning; sparse vectors (BGE-M3 lexical weights)
capture exact keyword / number matches — the same hybrid idea as the old
LanceDB+BM25 setup, but the lexical side now rides on BGE-M3's own sparse output.

We run a dense search and a sparse search, then fuse them with Reciprocal Rank
Fusion (RRF) in Python — robust and identical in spirit to the previous store,
and works in Qdrant's local (no-server) mode.
"""

from typing import Any, Dict, List, Tuple

from qdrant_client import QdrantClient, models

SparseVec = Tuple[List[int], List[float]]


class QdrantStore:
    DENSE = "dense"
    SPARSE = "sparse"

    def __init__(self, db_path: str, collection_name: str, embedding_dim: int,
                 url: str = ""):
        # url set  -> connect to a Qdrant server (docker-compose service).
        # url empty -> embedded mode: data in a local folder, no server.
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(path=db_path)
        self.collection = collection_name
        self.embedding_dim = embedding_dim

    def create_or_open(self):
        if not self.client.collection_exists(self.collection):
            try:
                self.client.create_collection(
                    self.collection,
                    vectors_config={
                        self.DENSE: models.VectorParams(
                            size=self.embedding_dim,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={self.SPARSE: models.SparseVectorParams()},
                )
            except Exception:
                # Another process (e.g. the API and a CLI ingest) may create it
                # first — a 409 "already exists" is fine, just proceed.
                if not self.client.collection_exists(self.collection):
                    raise
        return self

    def add(self, records: List[Dict[str, Any]]) -> None:
        """
        records: list of dicts with keys
            id (str/uuid), text, source, metadata (str),
            dense (list[float]), sparse ((indices, values))
        """
        points = []
        for r in records:
            idx, val = r["sparse"]
            points.append(
                models.PointStruct(
                    id=r["id"],
                    vector={
                        self.DENSE: r["dense"],
                        self.SPARSE: models.SparseVector(indices=idx, values=val),
                    },
                    payload={
                        "text": r["text"],
                        "source": r["source"],
                        "metadata": r["metadata"],
                        "seq": r.get("seq", -1),        # position within its file
                        "section": r.get("section", ""), # e.g. "المادة 17" (for filtering)
                    },
                )
            )
        self.client.upsert(self.collection, points=points, wait=True)

    def _payload_item(self, p) -> Dict:
        pl = p.payload or {}
        return {
            "id": p.id,
            "text": pl.get("text", ""),
            "source": pl.get("source", "unknown"),
            "metadata": pl.get("metadata", "{}"),
            "seq": pl.get("seq", -1),
            "section": pl.get("section", ""),
        }

    def search_section(self, section: str, top_k: int = 5) -> List[Dict]:
        """Return chunks whose payload.section matches (exact article boost)."""
        try:
            flt = models.Filter(must=[models.FieldCondition(
                key="section", match=models.MatchValue(value=section))])
            points, _ = self.client.scroll(
                self.collection, scroll_filter=flt, limit=top_k,
                with_payload=True, with_vectors=False)
            return [self._payload_item(p) for p in points]
        except Exception:
            return []

    def fetch_neighbors(self, source: str, seq: int, window: int = 1) -> List[Dict]:
        """Return chunks from the same file with seq within +/- window."""
        if seq is None or seq < 0:
            return []
        try:
            flt = models.Filter(must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=source)),
                models.FieldCondition(key="seq", range=models.Range(
                    gte=seq - window, lte=seq + window)),
            ])
            points, _ = self.client.scroll(
                self.collection, scroll_filter=flt, limit=2 * window + 1,
                with_payload=True, with_vectors=False)
            return [self._payload_item(p) for p in points]
        except Exception:
            return []

    # Indexes are maintained automatically by Qdrant; kept for interface parity.
    def build_fts_index(self, field: str = "text") -> None:
        return None

    def build_vector_index(self) -> bool:
        return False

    # --- retrieval ---

    def _search(self, vec, using, top_k):
        return self.client.query_points(
            self.collection, query=vec, using=using, limit=top_k, with_payload=True
        ).points

    def hybrid_search(
        self,
        query_dense,
        query_sparse: SparseVec,
        top_k: int = 20,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> List[Dict]:
        dense_hits = self._search(list(query_dense), self.DENSE, top_k)
        idx, val = query_sparse
        sparse_hits = (
            self._search(models.SparseVector(indices=idx, values=val), self.SPARSE, top_k)
            if idx
            else []
        )

        scores: Dict[Any, float] = {}
        items: Dict[Any, Dict] = {}

        def fold(hits, weight):
            for rank, p in enumerate(hits, start=1):
                scores[p.id] = scores.get(p.id, 0.0) + weight / (rrf_k + rank)
                if p.id not in items:
                    payload = p.payload or {}
                    items[p.id] = {
                        "id": p.id,
                        "text": payload.get("text", ""),
                        "source": payload.get("source", "unknown"),
                        "metadata": payload.get("metadata", "{}"),
                    }

        fold(dense_hits, vector_weight)
        fold(sparse_hits, bm25_weight)

        ranked = sorted(scores, key=scores.get, reverse=True)
        return [{"rrf_score": scores[i], **items[i]} for i in ranked[:top_k]]

    # --- stats helpers (used by the API) ---

    def count(self) -> int:
        try:
            return self.client.count(self.collection, exact=True).count
        except Exception:
            return 0

    def distinct_sources(self) -> int:
        """Number of distinct source files. Uses facet when available."""
        try:
            res = self.client.facet(self.collection, key="source", limit=100000)
            return len(res.hits)
        except Exception:
            pass
        # Fallback: scroll source payloads only.
        try:
            seen, offset = set(), None
            while True:
                points, offset = self.client.scroll(
                    self.collection, limit=2048, offset=offset,
                    with_payload=["source"], with_vectors=False,
                )
                for p in points:
                    seen.add((p.payload or {}).get("source"))
                if offset is None:
                    break
            return len(seen)
        except Exception:
            return 0

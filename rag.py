"""
End-to-end RAG pipeline.

Flow (query):
    user query
      → embed
      → hybrid search (vector + BM25 with RRF) → top-20 candidates
      → cross-encoder rerank → top-5
      → LLM generates answer grounded in the top-5

Flow (ingest):
    text → recursive chunking → batch embed → store in LanceDB
    (call build_indexes() once after all documents are ingested)
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Silence the harmless "XLMRobertaTokenizerFast ... use __call__" notices that
# transformers prints when FlagEmbedding loads the embedder/reranker tokenizers.
from transformers.utils import logging as _hf_logging
_hf_logging.set_verbosity_error()

from config import RAGConfig
from chunking import RecursiveChunker
from embeddings import BGEEmbedder
from reranker import BGEReranker
from vector_store import HybridStore
from llm import OllamaLLM


class RAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        self.chunker = RecursiveChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.embedder = BGEEmbedder(
            model_name=self.config.embedding_model,
            use_fp16=self.config.embedding_use_fp16,
            device=self.config.embedding_device,
        )
        self.reranker = BGEReranker(
            model_name=self.config.reranker_model,
            use_fp16=self.config.reranker_use_fp16,
            device=self.config.reranker_device,
        )
        self.store = HybridStore(
            db_path=self.config.db_path,
            table_name=self.config.table_name,
            embedding_dim=self.config.embedding_dim,
        )
        self.store.create_or_open()
        self.llm = OllamaLLM(
            model=self.config.llm_model,
            base_url=self.config.ollama_base_url,
        )

    

    def ingest_text(
        self,
        text: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        chunks = self.chunker.split_text(text)
        if not chunks:
            return 0

        vectors = self.embedder.embed_documents(
            chunks,
            batch_size=self.config.embedding_batch_size,
        )

        records = [
            {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": source,
                "metadata": json.dumps(metadata or {}, ensure_ascii=False),
                "vector": vec.tolist(),
            }
            for chunk, vec in zip(chunks, vectors)
        ]
        self.store.add(records)
        return len(records)

    def ingest_file(self, file_path: str) -> int:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            text = self._read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(
            text,
            source=str(path),
            metadata={"filename": path.name},
        )

    @staticmethod
    def _read_pdf(path: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages)

    def build_indexes(self) -> None:
        """Call once after ingesting all documents to enable BM25 search."""
        self.store.build_fts_index("text")

    # ---------- Retrieval ----------

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)

        # Stage 1: hybrid search → top initial_top_k candidates
        candidates = self.store.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            top_k=self.config.initial_top_k,
            vector_weight=self.config.vector_weight,
            bm25_weight=self.config.bm25_weight,
            rrf_k=self.config.rrf_k,
        )
        if not candidates:
            return []

        # Stage 2: rerank → top final_top_k
        texts = [c["text"] for c in candidates]
        reranked = self.reranker.rerank(
            query=query,
            documents=texts,
            top_k=self.config.final_top_k,
        )

        return [
            {**candidates[idx], "rerank_score": float(score)}
            for idx, score in reranked
        ]

    # ---------- Generation ----------

    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        retrieved = self.retrieve(question)
        contexts = [r["text"] for r in retrieved]
        answer = self.llm.generate(
            query=question,
            contexts=contexts,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        result = {"answer": answer}
        if return_sources:
            result["sources"] = retrieved
        return result

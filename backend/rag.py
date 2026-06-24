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

        # If the config asks for CUDA but there's no working CUDA build of
        # torch, fall back to CPU instead of crashing on model load.
        embed_device = self._resolve_device(self.config.embedding_device)
        rerank_device = self._resolve_device(self.config.reranker_device)
        embed_fp16 = self.config.embedding_use_fp16 and embed_device != "cpu"
        rerank_fp16 = self.config.reranker_use_fp16 and rerank_device != "cpu"

        self.chunker = RecursiveChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.embedder = BGEEmbedder(
            model_name=self.config.embedding_model,
            use_fp16=embed_fp16,
            device=embed_device,
        )
        self.reranker = BGEReranker(
            model_name=self.config.reranker_model,
            use_fp16=rerank_fp16,
            device=rerank_device,
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

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    print(
                        f"[warn] device '{device}' requested but CUDA is not "
                        f"available; falling back to CPU. Install a CUDA build "
                        f"of torch for GPU acceleration.",
                        file=__import__("sys").stderr,
                    )
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    # Extensions we know how to read.
    SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".txt", ".md", ".csv"}

    def ingest_file(self, file_path: str) -> int:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._read_pdf(path)
        elif suffix in (".xlsx", ".xls"):
            text = self._read_excel(path)
        elif suffix == ".csv":
            text = self._read_csv(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(
            text,
            source=str(path),
            metadata={"filename": path.name},
        )

    @staticmethod
    def _read_pdf(path: Path) -> str:
        # PyMuPDF (fitz) is ~5-10x faster than pypdf on large PDFs. Fall back
        # to pypdf if it isn't installed.
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
            return "\n\n".join(pages)

    @staticmethod
    def _read_excel(path: Path) -> str:
        """Flatten every sheet to text: 'col: value | col: value' per row."""
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None, dtype=str, engine=None)
        parts: List[str] = []
        for sheet_name, df in sheets.items():
            df = df.fillna("")
            parts.append(f"### Sheet: {sheet_name}")
            for _, row in df.iterrows():
                cells = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)

    @staticmethod
    def _read_csv(path: Path) -> str:
        import pandas as pd

        df = pd.read_csv(path, dtype=str).fillna("")
        rows = [
            " | ".join(f"{col}: {val}" for col, val in row.items() if str(val).strip())
            for _, row in df.iterrows()
        ]
        return "\n".join(rows)

    def build_indexes(self) -> None:
        """
        Call once after ingesting all documents. Builds:
          * the BM25 (FTS) index for keyword search, and
          * an ANN vector index for fast semantic search on large corpora.
        """
        self.store.build_fts_index("text")
        built = self.store.build_vector_index()
        if not built:
            print("[info] Skipped ANN vector index (too few rows; brute-force "
                  "search is fine at this size).")

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

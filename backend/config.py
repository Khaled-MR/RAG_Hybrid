"""
Central configuration for the RAG pipeline.

Defaults are tuned for an RTX 5060 Ti (16 GB VRAM). Embedder + reranker both
run on the GPU; the LLM runs in Ollama (also GPU). 16 GB is plenty for all
three at once.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Absolute path to this backend folder, so paths work no matter what the
# current working directory is (CLI from root, API launched elsewhere, etc).
_BACKEND_DIR = Path(__file__).resolve().parent


@dataclass
class RAGConfig:

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cuda"         # set to "cpu" if no CUDA build of torch
    embedding_use_fp16: bool = True        # fp16 ~halves VRAM, faster on GPU
    embedding_batch_size: int = 64         # 16 GB can handle large batches

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    reranker_use_fp16: bool = True

    chunk_size: int = 500
    chunk_overlap: int = 80

    db_path: str = str(_BACKEND_DIR / "lancedb")
    table_name: str = "documents"

    # Folder you drop your files into. `python ingest.py` ingests everything
    # under here (recursively) with no extra arguments.
    data_dir: str = str(_BACKEND_DIR / "data")
    # File types picked up automatically when ingesting a folder.
    ingest_extensions: str = ".pdf,.xlsx,.xls,.txt,.md"

    initial_top_k: int = 40      # wider recall — GPU reranker handles it fast
    final_top_k: int = 6         # a bit more context for richer answers
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_k: int = 60

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    temperature: float = 0.1
    max_tokens: int = 1024

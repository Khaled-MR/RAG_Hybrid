"""
Central configuration for the RAG pipeline.

Defaults are tuned for an RTX 5060 Ti (16 GB VRAM). Embedder + reranker both
run on the GPU; the LLM runs in Ollama (also GPU). 16 GB is plenty for all
three at once.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Absolute path to this backend folder, so paths work no matter what the
# current working directory is (CLI from root, API launched elsewhere, etc).
_BACKEND_DIR = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    """Read an override from the environment (used by docker-compose)."""
    return os.getenv(name, default)


@dataclass
class RAGConfig:

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    # "auto" = GPU only if the card has plenty of VRAM (>=10GB), else CPU so the
    # LLM keeps the whole GPU. Force with "cuda" / "cpu".
    embedding_device: str = "auto"
    embedding_use_fp16: bool = True        # fp16 ~halves VRAM, faster on GPU
    embedding_batch_size: int = 64         # 16 GB can handle large batches

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"
    reranker_use_fp16: bool = True

    # Larger chunks keep a whole article / section / clause together so it can
    # be quoted verbatim and answered precisely. Structure-aware splitting
    # (see chunking.py) keeps each "المادة N" / heading as its own chunk.
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Qdrant vector store (hybrid search).
    #   - QDRANT_URL set  -> connect to a Qdrant server (docker-compose service)
    #   - QDRANT_URL empty -> embedded local file mode (single process)
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", ""))
    qdrant_path: str = str(_BACKEND_DIR / "qdrant_db")
    collection_name: str = "documents"
    # (legacy LanceDB paths, kept so old code/data still resolves)
    db_path: str = str(_BACKEND_DIR / "lancedb")
    table_name: str = "documents"

    # Folder you drop your files into. `python ingest.py` ingests everything
    # under here (recursively) with no extra arguments.
    data_dir: str = str(_BACKEND_DIR / "data")
    # File types picked up automatically when ingesting a folder.
    ingest_extensions: str = ".pdf,.xlsx,.xls,.txt,.md"

    initial_top_k: int = 40      # wider recall — GPU reranker handles it fast
    final_top_k: int = 6         # a bit more context for richer answers

    # Quality: rewrite colloquial questions into a formal query before retrieval
    # (one extra fast LLM call — improves recall on messy/colloquial questions).
    # It's also history-aware: turns a follow-up into a standalone question.
    enable_query_rewrite: bool = True
    # Retrieval-quality boosters (all retrieval-side → negligible latency):
    enable_multi_query: bool = True        # search original + rewritten, RRF-merge
    enable_article_filter: bool = True      # boost the exact "المادة N" if asked
    enable_neighbor_expansion: bool = True  # add neighbouring chunks for fuller context
    neighbor_window: int = 1                # how many chunks each side to add
    max_context_passages: int = 12          # cap total passages sent to the LLM (speed)
    history_turns: int = 4                  # past turns used for rewrite + answer
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_k: int = 60

    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    # Qwen family (LLM). qwen2.5:7b fits alongside BGE on 16GB; for stronger
    # Arabic on 16GB use "qwen2.5:14b". Override with LLM_MODEL env in docker.
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen2.5:7b"))
    temperature: float = 0.0     # deterministic — least hallucination for factual QA
    max_tokens: int = 4096        # cap on answer length; lower = faster generation
    # Context window. Big enough for the prompt (~900 tok) + answer, small enough
    # to not waste VRAM. Raise if you increase final_top_k or chunk_size a lot.
    num_ctx: int = 8192
    # Keep the model loaded in VRAM between questions ("0" = unload immediately,
    # "30m" = 30 min, "-1" = forever). Avoids a 5-10s reload per question.
    llm_keep_alive: str = "30m"


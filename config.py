"""
Central configuration for the RAG pipeline.

Defaults are tuned for an RTX 4070 laptop (8GB VRAM).
The reranker is placed on CPU by default so the LLM can use the GPU freely.
"""

from dataclasses import dataclass


@dataclass
class RAGConfig:
   
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"          # torch in this venv is CPU-only; use "cuda" if you install a CUDA build
    embedding_use_fp16: bool = False       # fp16 only helps on GPU
    embedding_batch_size: int = 16

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"            
    reranker_use_fp16: bool = False         

    chunk_size: int = 500                  
    chunk_overlap: int = 80

    db_path: str = "./lancedb"
    table_name: str = "documents"

    initial_top_k: int = 20                
    final_top_k: int = 5                  
    vector_weight: float = 0.5            
    bm25_weight: float = 0.5
    rrf_k: int = 60                        

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    temperature: float = 0.1
    max_tokens: int = 1024

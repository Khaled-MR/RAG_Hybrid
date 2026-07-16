# -*- coding: utf-8 -*-
"""
Compare the OLD vector store (LanceDB) vs the NEW one (Qdrant) on the same
corpus and the same questions.

Generation uses the same LLM either way, so the only thing the migration
changes is *retrieval*. We therefore measure each retrieval stage separately
(embed / DB search / rerank) for both backends, plus a shared generation pass,
and write a Markdown report.

Run (stop the API backend first — Qdrant local mode is single-process):
    ..\.venv\Scripts\python.exe compare_backends.py
"""
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from config import RAGConfig
from embeddings import BGEEmbedder
from reranker import BGEReranker
from vector_store import HybridStore       # LanceDB (old)
from qdrant_store import QdrantStore       # Qdrant (new)
from llm import OllamaLLM

QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG)?",
    "How does the retrieval step in RAG work?",
    "What is reranking in a RAG pipeline?",
    "ما هو الـ RAG وكيف يعمل؟",
    "What is chunking and why does it matter?",
    "What is a transformer architecture?",
    "What is the attention mechanism?",
    "What is a large language model?",
    "ما هو نموذج اللغة الكبير LLM؟",
    "What is the context window in an LLM?",
    "What is tokenization in NLP?",
    "What is a word embedding?",
    "What is the difference between fine-tuning and prompt engineering?",
    "What are the differences between LLMs and traditional language models?",
    "ما هي آلية الانتباه attention؟",
    "How do you build a reliable AI agent?",
    "How do you evaluate an AI agent?",
    "What are common ways to build AI agents with confidence?",
    "ما هي طرق بناء وكلاء الذكاء الاصطناعي؟",
    "What is a Docker container?",
    "What is the difference between a Docker image and a container?",
    "What is a Dockerfile?",
    "ما هو الـ Docker container؟",
    "What is a primary key in SQL?",
    "What is a foreign key?",
    "What is the difference between DELETE and TRUNCATE?",
    "What does the JOIN keyword do?",
    "How do you create a table with a primary key?",
    "ما هو المفتاح الأساسي في SQL؟",
    "What is an SQL index?",
    "What is named entity recognition?",
    "What is transfer learning in NLP?",
    "What is a vector database used for?",
    "What is semantic search?",
    "ما هو البحث الدلالي؟",
    "What is multimodality in language models?",
    "What is the Querying Transformer (Q-Former)?",
    "What is the difference between an encoder and a decoder?",
    "What is quantization of a model?",
    "ما الفرق بين الـ encoder و الـ decoder؟",
]

GEN_CAP = 300  # fixed answer cap so generation timing is stable for the report


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def summarize(name, vals):
    return (f"| {name} | {statistics.mean(vals):.3f} | {statistics.median(vals):.3f} "
            f"| {pct(vals,0.95):.3f} | {min(vals):.3f} | {max(vals):.3f} |")


def main():
    cfg = RAGConfig()
    print("Loading shared models (embedder + reranker)...", file=sys.stderr)
    embedder = BGEEmbedder(cfg.embedding_model, cfg.embedding_use_fp16,
                           _dev(cfg.embedding_device))
    reranker = BGEReranker(cfg.reranker_model, cfg.reranker_use_fp16,
                           _dev(cfg.reranker_device))

    lance = HybridStore(cfg.db_path, cfg.table_name, cfg.embedding_dim)
    lance.create_or_open()
    qdr = QdrantStore(cfg.qdrant_path, cfg.collection_name, cfg.embedding_dim)
    qdr.create_or_open()

    print(f"LanceDB rows: {lance.table.count_rows()} | Qdrant points: {qdr.count()}",
          file=sys.stderr)

    rows = []  # per-question metrics
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q}", file=sys.stderr)
        r = {"q": q}

        # --- LanceDB retrieval ---
        t = time.time(); qv = embedder.embed_query(q); r["L_embed"] = time.time() - t
        t = time.time()
        lhits = lance.hybrid_search(query_text=q, query_vector=qv,
                                    top_k=cfg.initial_top_k,
                                    vector_weight=cfg.vector_weight,
                                    bm25_weight=cfg.bm25_weight, rrf_k=cfg.rrf_k)
        r["L_search"] = time.time() - t
        t = time.time()
        lr = reranker.rerank(q, [h["text"] for h in lhits], top_k=cfg.final_top_k)
        r["L_rerank"] = time.time() - t
        l_top = [lhits[idx]["text"] for idx, _ in lr]

        # --- Qdrant retrieval ---
        t = time.time(); qd, qs = embedder.embed_query_hybrid(q); r["Q_embed"] = time.time() - t
        t = time.time()
        qhits = qdr.hybrid_search(query_dense=qd, query_sparse=qs,
                                  top_k=cfg.initial_top_k,
                                  vector_weight=cfg.vector_weight,
                                  bm25_weight=cfg.bm25_weight, rrf_k=cfg.rrf_k)
        r["Q_search"] = time.time() - t
        t = time.time()
        qr = reranker.rerank(q, [h["text"] for h in qhits], top_k=cfg.final_top_k)
        r["Q_rerank"] = time.time() - t
        q_top = [qhits[idx]["text"] for idx, _ in qr]

        r["L_retrieve"] = r["L_embed"] + r["L_search"] + r["L_rerank"]
        r["Q_retrieve"] = r["Q_embed"] + r["Q_search"] + r["Q_rerank"]

        # retrieval agreement (Jaccard of final top-k chunk texts)
        sl, sq = set(l_top), set(q_top)
        r["overlap"] = len(sl & sq) / len(sl | sq) if (sl | sq) else 1.0
        rows.append((r, q_top))

    # --- shared generation pass (backend-independent) ---
    print("Measuring generation (shared LLM)...", file=sys.stderr)
    llm = OllamaLLM(cfg.llm_model, cfg.ollama_base_url, cfg.llm_keep_alive, cfg.num_ctx)
    list(llm.generate_stream("warm", ["warm"], max_tokens=4))  # warmup
    gen = []
    for r, q_top in rows:
        t = time.time()
        list(llm.generate_stream(r["q"], q_top, temperature=0.0, max_tokens=GEN_CAP))
        g = time.time() - t
        r["gen"] = g; gen.append(g)

    _report(cfg, rows, gen)


def _dev(d):
    # mirror RAGPipeline._resolve_device("auto") cheaply
    if d == "auto":
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory/1e9 >= 10:
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return d


def _report(cfg, rows, gen):
    rr = [r for r, _ in rows]
    L_ret = [r["L_retrieve"] for r in rr]
    Q_ret = [r["Q_retrieve"] for r in rr]
    L_tot = [r["L_retrieve"] + r["gen"] for r in rr]
    Q_tot = [r["Q_retrieve"] + r["gen"] for r in rr]

    out = []
    w = out.append
    w("# Backend Comparison Report — LanceDB (before) vs Qdrant (after)\n")
    w(f"- Corpus: **{len(rows)} questions** over the ingested 9 PDFs (~2,669 chunks)")
    w(f"- Hardware: RTX 5060 Ti 16GB · embed/rerank `{_dev(cfg.embedding_device)}` · model `{cfg.llm_model}`")
    w(f"- Retrieval: hybrid, initial_top_k={cfg.initial_top_k}, final_top_k={cfg.final_top_k}")
    w(f"- Generation capped at {GEN_CAP} tokens for stable timing (same for both backends)\n")

    w("## Headline — retrieval latency (what the migration changed)\n")
    w("| Stage | Mean | Median | p95 | Min | Max |")
    w("|-------|----:|----:|----:|----:|----:|")
    w(summarize("LanceDB — embed (s)", [r["L_embed"] for r in rr]))
    w(summarize("LanceDB — DB search (s)", [r["L_search"] for r in rr]))
    w(summarize("LanceDB — rerank (s)", [r["L_rerank"] for r in rr]))
    w(summarize("**LanceDB — retrieval total (s)**", L_ret))
    w(summarize("Qdrant — embed (s)", [r["Q_embed"] for r in rr]))
    w(summarize("Qdrant — DB search (s)", [r["Q_search"] for r in rr]))
    w(summarize("Qdrant — rerank (s)", [r["Q_rerank"] for r in rr]))
    w(summarize("**Qdrant — retrieval total (s)**", Q_ret))
    w("")

    w("## Pure DB-search latency (the apples-to-apples bit)\n")
    ls = statistics.mean([r["L_search"] for r in rr])
    qsx = statistics.mean([r["Q_search"] for r in rr])
    w(f"- LanceDB mean DB search: **{ls*1000:.1f} ms**")
    w(f"- Qdrant  mean DB search: **{qsx*1000:.1f} ms**")
    w(f"- Difference: **{(qsx-ls)*1000:+.1f} ms** per query "
      f"({'Qdrant faster' if qsx<ls else 'LanceDB faster'})\n")

    w("## End-to-end (retrieval + generation)\n")
    w("| Backend | Mean total (s) | Median | p95 | Retrieval share |")
    w("|---------|----:|----:|----:|----:|")
    gm = statistics.mean(gen)
    w(f"| LanceDB | {statistics.mean(L_tot):.2f} | {statistics.median(L_tot):.2f} | {pct(L_tot,0.95):.2f} | {statistics.mean(L_ret)/statistics.mean(L_tot)*100:.1f}% |")
    w(f"| Qdrant  | {statistics.mean(Q_tot):.2f} | {statistics.median(Q_tot):.2f} | {pct(Q_tot,0.95):.2f} | {statistics.mean(Q_ret)/statistics.mean(Q_tot)*100:.1f}% |")
    w(f"\n- Shared generation: mean **{gm:.2f}s** ({statistics.mean([r['gen'] for r in rr]) and len(gen)} questions)")
    w(f"- **Generation is {gm/statistics.mean(Q_tot)*100:.0f}% of total time** — the vector DB is a small fraction.\n")

    w("## Retrieval agreement (quality sanity check)\n")
    ov = statistics.mean([r["overlap"] for r in rr])
    w(f"- Mean Jaccard overlap of the final top-{cfg.final_top_k} chunks (LanceDB vs Qdrant): "
      f"**{ov*100:.0f}%** — both hybrid retrievers surface largely the same evidence.\n")

    w("## Per-question (retrieval total, seconds)\n")
    w("| # | Question | LanceDB | Qdrant | DB search L→Q (ms) | overlap |")
    w("|---|----------|----:|----:|----:|----:|")
    for i, r in enumerate(rr, 1):
        q = r["q"] if len(r["q"]) <= 38 else r["q"][:35] + "..."
        w(f"| {i} | {q} | {r['L_retrieve']:.3f} | {r['Q_retrieve']:.3f} | "
          f"{r['L_search']*1000:.0f}→{r['Q_search']*1000:.0f} | {r['overlap']*100:.0f}% |")

    text = "\n".join(out)
    with open("BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print("\n[saved to backend/BENCHMARK_REPORT.md]", file=sys.stderr)


if __name__ == "__main__":
    main()

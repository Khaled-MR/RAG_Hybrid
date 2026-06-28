# -*- coding: utf-8 -*-
"""
Benchmark the RAG pipeline against the ingested documents.

Asks a list of questions, measures per-question latency / throughput, and
prints a Markdown table plus aggregate stats (mean, p50, p95).

Run from the backend/ folder:
    ..\.venv\Scripts\python.exe benchmark.py
"""
import sys
import time
import statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from rag import RAGPipeline
from config import RAGConfig

QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG) and how does it work?",
    "What is a Docker container and how is it different from an image?",
    "What is a transformer architecture in large language models?",
    "How do you build a reliable AI agent?",
    "How do you evaluate the performance of an AI agent?",
    "What is tokenization in natural language processing?",
    "What is the difference between fine-tuning and prompt engineering?",
    "What is a vector embedding and why is it useful for search?",
    "ما هو الـ RAG ولماذا يُستخدم؟",
    "What is a primary key in SQL?",
]


def bench_one(rag: RAGPipeline, question: str) -> dict:
    # --- retrieval ---
    t0 = time.time()
    retrieved = rag.retrieve(question)
    retrieve_s = time.time() - t0
    contexts = [r["text"] for r in retrieved]

    # --- generation (stream so we can measure time-to-first-token) ---
    messages = rag.llm._build_messages(question, contexts, None)
    stream = rag.llm.client.chat(
        model=rag.llm.model,
        messages=messages,
        keep_alive=rag.llm.keep_alive,
        stream=True,
        options={
            "temperature": rag.config.temperature,
            "num_predict": rag.config.max_tokens,
            "num_ctx": rag.config.num_ctx,
        },
    )
    g0 = time.time()
    ttft = None
    last = None
    for part in stream:
        if ttft is None and part.get("message", {}).get("content"):
            ttft = time.time() - g0
        last = part
    gen_s = time.time() - g0
    total_s = retrieve_s + gen_s

    prompt_tok = (last or {}).get("prompt_eval_count", 0)
    out_tok = (last or {}).get("eval_count", 0)
    eval_dur = (last or {}).get("eval_duration", 0) / 1e9
    gen_tok_s = out_tok / eval_dur if eval_dur else 0.0

    return {
        "q": question,
        "retrieve_s": retrieve_s,
        "ttft_s": ttft or gen_s,
        "gen_s": gen_s,
        "total_s": total_s,
        "prompt_tok": prompt_tok,
        "out_tok": out_tok,
        "gen_tok_s": gen_tok_s,
    }


def main():
    cfg = RAGConfig()
    print("Loading pipeline...", file=sys.stderr)
    rag = RAGPipeline(cfg)

    # warm up the LLM (load into VRAM) so the first row isn't an outlier
    print("Warming up...", file=sys.stderr)
    list(rag.llm.generate_stream("warmup", ["warmup"], max_tokens=4))

    rows = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q}", file=sys.stderr)
        rows.append(bench_one(rag, q))

    # --- per-question table ---
    print(f"\n## Setup\n")
    print(f"- Model: `{cfg.llm_model}`  |  embed/rerank device: "
          f"`{rag._resolve_device(cfg.embedding_device)}`  |  temp: {cfg.temperature}")
    print(f"- Retrieval: hybrid (vector+BM25) → rerank, "
          f"initial_top_k={cfg.initial_top_k}, final_top_k={cfg.final_top_k}")
    print(f"- max_tokens={cfg.max_tokens}, num_ctx={cfg.num_ctx}\n")

    print("## Per-question results\n")
    print("| # | Question | Retrieve (s) | TTFT (s) | Generate (s) | "
          "Total (s) | In tok | Out tok | Gen tok/s |")
    print("|---|----------|----:|----:|----:|----:|----:|----:|----:|")
    for i, r in enumerate(rows, 1):
        q = r["q"] if len(r["q"]) <= 42 else r["q"][:39] + "..."
        print(f"| {i} | {q} | {r['retrieve_s']:.2f} | {r['ttft_s']:.2f} | "
              f"{r['gen_s']:.2f} | {r['total_s']:.2f} | {r['prompt_tok']} | "
              f"{r['out_tok']} | {r['gen_tok_s']:.1f} |")

    # --- aggregates ---
    def agg(key):
        vals = [r[key] for r in rows]
        return statistics.mean(vals), statistics.median(vals), max(vals)

    tot = [r["total_s"] for r in rows]
    p95 = sorted(tot)[int(len(tot) * 0.95) - 1]
    total_out = sum(r["out_tok"] for r in rows)
    total_time = sum(tot)

    print("\n## Aggregate metrics\n")
    print("| Metric | Mean | Median (p50) | Max |")
    print("|--------|----:|----:|----:|")
    for label, key in [
        ("Retrieve latency (s)", "retrieve_s"),
        ("Time to first token (s)", "ttft_s"),
        ("Generation time (s)", "gen_s"),
        ("Total latency (s)", "total_s"),
        ("Generation throughput (tok/s)", "gen_tok_s"),
    ]:
        m, md, mx = agg(key)
        print(f"| {label} | {m:.2f} | {md:.2f} | {mx:.2f} |")

    print(f"\n- **p95 total latency:** {p95:.2f} s")
    print(f"- **End-to-end throughput:** {total_out / total_time:.1f} output tok/s "
          f"across {len(rows)} questions ({total_out} tokens in {total_time:.1f}s)")
    print(f"- **Questions/min (sequential):** {60 * len(rows) / total_time:.1f}")


if __name__ == "__main__":
    main()

"""
Quick test harness for the RAG pipeline.

Loads the models once, then runs a list of questions against the ingested
documents and prints each answer with its sources. Use this to sanity-check
retrieval/answer quality from code instead of the query.py CLI.

Run:
    .\.venv\Scripts\python.exe test_rag.py

Edit the QUESTIONS list below to try your own questions, or pass one on the
command line:
    .\.venv\Scripts\python.exe test_rag.py "What is a foreign key?"
"""

import sys
import time

# Windows consoles default to cp1252, which chokes on some PDF glyphs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag import RAGPipeline


# --- edit these to test whatever you want ---
QUESTIONS = [
    "What is a primary key in SQL?",
    "How does the SELECT statement work?",
    "What is the difference between DELETE and TRUNCATE?",
    "ما هو الـ primary key في SQL؟",   # Arabic — should answer in Arabic
]


def ask(rag: RAGPipeline, question: str) -> None:
    print("=" * 70)
    print(f"Q: {question}")
    print("-" * 70)

    t0 = time.time()
    result = rag.query(question)
    elapsed = time.time() - t0

    print("Answer:")
    print(result["answer"])

    print("\nSources:")
    for i, src in enumerate(result.get("sources", []), 1):
        score = src.get("rerank_score", 0.0)
        source = src.get("source", "unknown")
        preview = src["text"][:100].replace("\n", " ")
        print(f"  [{i}] (rerank={score:.3f}) {source}")
        print(f"      {preview}...")

    print(f"\n(took {elapsed:.1f}s)\n")


def main():
    print("Loading RAG pipeline (embedder + reranker + LLM)...")
    t0 = time.time()
    rag = RAGPipeline()
    print(f"Ready in {time.time() - t0:.1f}s\n")

    # If a question is passed on the command line, use that; otherwise run the list.
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else QUESTIONS

    for q in questions:
        ask(rag, q)


if __name__ == "__main__":
    main()

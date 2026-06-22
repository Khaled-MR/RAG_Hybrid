"""
Query the RAG pipeline.

Usage:
    python query.py "What is the policy on X?"
    python query.py --interactive
"""

import argparse
import sys

from rag import RAGPipeline

# Windows consoles default to cp1252, which chokes on some Unicode glyphs
# carried over from PDFs (e.g. Wingdings bullets). Force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_result(result: dict) -> None:
    print("\nAnswer:")
    print(result["answer"])
    print("\n--- Sources ---")
    for i, src in enumerate(result.get("sources", []), 1):
        score = src.get("rerank_score", 0.0)
        source = src.get("source", "unknown")
        preview = src["text"][:120].replace("\n", " ")
        print(f"[{i}] (rerank={score:.3f}) {source}")
        print(f"     {preview}...")


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("-i", "--interactive", action="store_true")
    args = parser.parse_args()

    rag = RAGPipeline()

    if args.interactive or not args.question:
        print("RAG Q&A — type 'exit' or Ctrl-C to quit\n")
        try:
            while True:
                q = input("Q: ").strip()
                if q.lower() in {"exit", "quit"} or not q:
                    break
                result = rag.query(q)
                print_result(result)
                print()
        except (KeyboardInterrupt, EOFError):
            print()
    else:
        result = rag.query(args.question)
        print_result(result)


if __name__ == "__main__":
    main()

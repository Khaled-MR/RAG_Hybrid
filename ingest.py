
import argparse
import sys
from pathlib import Path

from rag import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--ext",
        default=".txt,.md",
        help="Comma-separated file extensions to include (default: .txt,.md)",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip rebuilding the FTS index at the end",
    )
    args = parser.parse_args()

    rag = RAGPipeline()
    path = Path(args.path)

    if not path.exists():
        print(f"[!] Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    extensions = [e.strip() for e in args.ext.split(",") if e.strip()]

    if path.is_file():
        files = [path]
    else:
        files = []
        for ext in extensions:
            files.extend(path.rglob(f"*{ext}"))

    if not files:
        print("[!] No matching files found.", file=sys.stderr)
        sys.exit(1)

    total = 0
    for f in files:
        try:
            n = rag.ingest_file(str(f))
            print(f"[+] {f.name}: {n} chunks")
            total += n
        except Exception as exc:
            print(f"[!] {f.name}: {exc}", file=sys.stderr)

    print(f"\nIngested {total} chunks total.")

    if not args.no_index:
        print("Building FTS (BM25) index...")
        rag.build_indexes()
        print("Done.")


if __name__ == "__main__":
    main()

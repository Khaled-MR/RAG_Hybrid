"""
Bulk ingestion for the RAG store — designed for tens of thousands of files.

Simplest usage ("just drop files in and run"):

    1. Put your PDFs / Excel / text files anywhere under  ./data
    2. Run:   .\.venv\Scripts\python.exe ingest.py

That's it. It walks ./data recursively, ingests every supported file
(.pdf .xlsx .xls .csv .txt .md), and builds the BM25 index at the end.

Key features for large batches (e.g. 43k files):
  * Resumable — keeps a manifest of finished files. Re-run after a crash /
    Ctrl-C and it skips everything already done.
  * Per-file error isolation — one corrupt PDF won't stop the run; failures
    are logged to ingest_errors.log and skipped.
  * Progress + ETA printed as it goes.

Other usage:
    python ingest.py "C:\\path\\to\\file.pdf"     # a single file
    python ingest.py "D:\\some\\folder"           # a specific folder
    python ingest.py --ext .pdf,.xlsx             # only these types
    python ingest.py --reset-manifest             # re-ingest everything
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 output so Arabic / odd PDF glyphs don't crash the console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag import RAGPipeline
from config import RAGConfig


def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def collect_files(root: Path, extensions: list[str]) -> list[Path]:
    if root.is_file():
        return [root]
    exts = {e.lower() for e in extensions}
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    config = RAGConfig()
    parser = argparse.ArgumentParser(description="Bulk-ingest documents into the RAG store")
    parser.add_argument(
        "path",
        nargs="?",
        default=config.data_dir,
        help=f"File or directory to ingest (default: {config.data_dir})",
    )
    parser.add_argument(
        "--ext",
        default=config.ingest_extensions,
        help=f"Comma-separated extensions (default: {config.ingest_extensions})",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip building the FTS (BM25) index at the end",
    )
    parser.add_argument(
        "--reset-manifest",
        action="store_true",
        help="Ignore the resume manifest and re-ingest every file",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Persist the manifest every N files (default: 50)",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"[!] Path not found: {root}", file=sys.stderr)
        if str(root) == config.data_dir:
            print(f"    Create it and drop your files in:  mkdir {config.data_dir}",
                  file=sys.stderr)
        sys.exit(1)

    extensions = [e.strip() for e in args.ext.split(",") if e.strip()]
    files = collect_files(root, extensions)
    if not files:
        print(f"[!] No matching files ({', '.join(extensions)}) under {root}",
              file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(config.db_path) / "ingested_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    error_log = Path("ingest_errors.log")

    manifest = {} if args.reset_manifest else load_manifest(manifest_path)

    # Figure out what still needs doing (resume: skip files unchanged since done).
    todo = []
    for f in files:
        key = str(f.resolve())
        try:
            mtime = f.stat().st_mtime
        except OSError:
            mtime = 0
        done = manifest.get(key)
        if done and abs(done.get("mtime", -1) - mtime) < 1:
            continue
        todo.append((f, key, mtime))

    skipped = len(files) - len(todo)
    print(f"Found {len(files)} files. Already done: {skipped}. To ingest: {len(todo)}.")
    if not todo:
        print("Nothing new to ingest.")
        if not args.no_index:
            print("Rebuilding FTS (BM25) index...")
            RAGPipeline(config).build_indexes()
            print("Done.")
        return

    print("Loading models (embedder on GPU)...")
    rag = RAGPipeline(config)

    total_chunks = 0
    failures = 0
    t_start = time.time()

    for i, (f, key, mtime) in enumerate(todo, 1):
        try:
            n = rag.ingest_file(str(f))
            manifest[key] = {"mtime": mtime, "chunks": n}
            total_chunks += n
            status = f"{n} chunks"
        except Exception as exc:
            failures += 1
            status = f"FAILED ({exc})"
            with error_log.open("a", encoding="utf-8") as log:
                log.write(f"{f}\t{exc}\n")

        elapsed = time.time() - t_start
        rate = i / elapsed if elapsed else 0
        eta = (len(todo) - i) / rate if rate else 0
        print(f"[{i}/{len(todo)}] {f.name}: {status}  "
              f"({rate:.1f} files/s, ETA {fmt_eta(eta)})")

        if i % args.save_every == 0:
            save_manifest(manifest_path, manifest)

    save_manifest(manifest_path, manifest)

    print(f"\nIngested {total_chunks} chunks from {len(todo) - failures} files "
          f"({failures} failed) in {fmt_eta(time.time() - t_start)}.")
    if failures:
        print(f"See {error_log} for failure details.")

    if not args.no_index:
        print("Building FTS (BM25) index...")
        rag.build_indexes()
        print("Done.")


if __name__ == "__main__":
    main()

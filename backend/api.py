"""
FastAPI server exposing the RAG pipeline to the web UI.

Endpoints:
    GET  /api/health          -> {status, ready}
    GET  /api/stats           -> {chunks, files}
    POST /api/ask             -> {answer, sources}   (body: {"question": "..."})
    POST /api/upload          -> {ingested, failed, chunks, files}  (multipart files)

Run:
    .\.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
    (run from the backend/ folder, or:  uvicorn backend.api:app ...)
"""

import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sibling modules importable whether launched as `api:app` (cwd=backend)
# or `backend.api:app` (cwd=project root).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAGConfig
from rag import RAGPipeline

config = RAGConfig()
DATA_DIR = Path(config.data_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG API")

# The Vite dev server runs on a different port, so allow cross-origin calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The pipeline (and its models) load lazily on first use — startup stays fast
# and importing this module for tests doesn't pull GPU models into memory.
_rag: RAGPipeline | None = None


def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline(config)
    return _rag


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    text: str
    source: str
    rerank_score: float = 0.0


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    elapsed: float


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": _rag is not None}


@app.get("/api/stats")
def stats():
    rag = get_rag()
    try:
        chunks = rag.store.table.count_rows()
    except Exception:
        chunks = 0
    # Count distinct source files. Projected scan (source column only, no
    # vectors) keeps this cheap even with millions of chunks.
    files = 0
    try:
        tbl = rag.store.table.to_lance().to_table(columns=["source"])
        files = len(set(tbl.column("source").to_pylist()))
    except Exception:
        try:
            tbl = rag.store.table.to_arrow()
            files = len(set(tbl.column("source").to_pylist()))
        except Exception:
            pass
    return {"chunks": chunks, "files": files}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    rag = get_rag()
    t0 = time.time()
    try:
        result = rag.query(question)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
    sources = [
        Source(
            text=s.get("text", ""),
            source=Path(s.get("source", "unknown")).name,
            rerank_score=float(s.get("rerank_score", 0.0)),
        )
        for s in result.get("sources", [])
    ]
    return AskResponse(answer=result["answer"], sources=sources, elapsed=time.time() - t0)


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Save uploaded files into data/, ingest each, and rebuild the indexes."""
    rag = get_rag()
    ingested, failed, total_chunks = 0, 0, 0
    results = []

    for uf in files:
        dest = DATA_DIR / Path(uf.filename).name
        try:
            with dest.open("wb") as out:
                shutil.copyfileobj(uf.file, out)
            n = rag.ingest_file(str(dest))
            ingested += 1
            total_chunks += n
            results.append({"file": dest.name, "chunks": n, "ok": True})
        except Exception as exc:
            failed += 1
            results.append({"file": uf.filename, "error": str(exc), "ok": False})
        finally:
            await uf.close()

    if ingested:
        try:
            rag.build_indexes()
        except Exception:
            traceback.print_exc()

    return {
        "ingested": ingested,
        "failed": failed,
        "chunks": total_chunks,
        "files": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

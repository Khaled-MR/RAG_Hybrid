// Thin client for the FastAPI RAG backend. Calls go through Vite's /api proxy.

export interface Source {
  text: string;
  source: string;
  rerank_score: number;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  elapsed: number;
}

export interface UploadResult {
  ingested: number;
  failed: number;
  chunks: number;
  files: { file: string; chunks?: number; error?: string; ok: boolean }[];
}

export interface Stats {
  chunks: number;
  files: number;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function ask(question: string): Promise<AskResponse> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return asJson<AskResponse>(res);
}

export async function uploadFiles(files: File[]): Promise<UploadResult> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch("/api/upload", { method: "POST", body: form });
  return asJson<UploadResult>(res);
}

export async function getStats(): Promise<Stats> {
  const res = await fetch("/api/stats");
  return asJson<Stats>(res);
}

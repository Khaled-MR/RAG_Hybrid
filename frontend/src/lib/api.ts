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

export interface StreamHandlers {
  onSources?: (sources: Source[]) => void;
  onDelta?: (text: string) => void;
  onDone?: (elapsed: number) => void;
}

/**
 * Ask with a streamed answer. Reads newline-delimited JSON from the backend
 * and invokes the handlers as events arrive. Resolves when the stream ends.
 */
export async function askStream(question: string, handlers: StreamHandlers): Promise<void> {
  const res = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handleLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const evt = JSON.parse(trimmed);
    if (evt.type === "sources") handlers.onSources?.(evt.sources);
    else if (evt.type === "delta") handlers.onDelta?.(evt.text);
    else if (evt.type === "done") handlers.onDone?.(evt.elapsed);
    else if (evt.type === "error") throw new Error(evt.detail);
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? ""; // keep the last partial line
    for (const line of lines) handleLine(line);
  }
  if (buffer.trim()) handleLine(buffer);
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

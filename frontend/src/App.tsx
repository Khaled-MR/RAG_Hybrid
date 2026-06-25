import React from "react";
import { Upload, Loader2, FileText, AlertCircle, ChevronDown, Database } from "lucide-react";
import { PromptInputBox } from "@/components/ui/ai-prompt-box";
import { askStream, uploadFiles, getStats, type Source, type Stats } from "@/lib/api";
import { cn } from "@/lib/utils";

const DOC_EXTENSIONS = ".pdf,.xlsx,.xls,.csv,.txt,.md";

// The exact gradient from the component demo.
const GRADIENT =
  "bg-[radial-gradient(125%_125%_at_50%_101%,rgba(245,87,2,1)_10.5%,rgba(245,120,2,1)_16%,rgba(245,140,2,1)_17.5%,rgba(245,170,100,1)_25%,rgba(238,174,202,1)_40%,rgba(202,179,214,1)_65%,rgba(148,201,233,1)_100%)]";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  elapsed?: number;
  error?: boolean;
}

interface Toast {
  id: string;
  kind: "info" | "success" | "error";
  text: string;
}

let idCounter = 0;
const uid = () => `${Date.now()}-${idCounter++}`;

export default function App() {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [isAsking, setIsAsking] = React.useState(false);
  const [isUploading, setIsUploading] = React.useState(false);
  const [stats, setStats] = React.useState<Stats | null>(null);
  const [toasts, setToasts] = React.useState<Toast[]>([]);
  const [dragging, setDragging] = React.useState(false);

  const docInputRef = React.useRef<HTMLInputElement>(null);
  const scrollRef = React.useRef<HTMLDivElement>(null);

  const pushToast = React.useCallback((kind: Toast["kind"], text: string) => {
    const id = uid();
    setToasts((t) => [...t, { id, kind, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }, []);

  const refreshStats = React.useCallback(async () => {
    try {
      setStats(await getStats());
    } catch {
      /* backend may still be warming up */
    }
  }, []);

  React.useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  React.useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isAsking]);

  const handleAsk = async (question: string) => {
    const q = question.trim();
    if (!q || isAsking) return;

    const answerId = uid();
    setMessages((m) => [
      ...m,
      { id: uid(), role: "user", content: q },
      { id: answerId, role: "assistant", content: "" },
    ]);
    setIsAsking(true);

    const patch = (fields: Partial<Message>) =>
      setMessages((m) => m.map((msg) => (msg.id === answerId ? { ...msg, ...fields } : msg)));

    try {
      await askStream(q, {
        onSources: (sources) => patch({ sources }),
        onDelta: (text) => setMessages((m) =>
          m.map((msg) => (msg.id === answerId ? { ...msg, content: msg.content + text } : msg))
        ),
        onDone: (elapsed) => patch({ elapsed }),
      });
    } catch (err) {
      patch({ content: `Couldn't get an answer: ${(err as Error).message}`, error: true });
    } finally {
      setIsAsking(false);
    }
  };

  const handleUpload = async (files: File[]) => {
    if (!files.length || isUploading) return;
    setIsUploading(true);
    pushToast("info", `Uploading ${files.length} file(s)…`);
    try {
      const res = await uploadFiles(files);
      if (res.ingested) {
        pushToast(
          "success",
          `Ingested ${res.ingested} file(s) → ${res.chunks} chunks.${res.failed ? ` ${res.failed} failed.` : ""}`
        );
      } else {
        pushToast("error", `No files ingested. ${res.failed} failed.`);
      }
      refreshStats();
    } catch (err) {
      pushToast("error", `Upload failed: ${(err as Error).message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const onWindowDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length) handleUpload(files);
  };

  const empty = messages.length === 0;

  return (
    <div
      className={cn("relative flex h-screen w-full flex-col items-center", GRADIENT)}
      onDragOver={(e) => {
        e.preventDefault();
        if (!dragging) setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.clientX === 0 && e.clientY === 0) setDragging(false);
      }}
      onDrop={onWindowDrop}
    >
      {/* Stats pill (top-right, glassy) */}
      {stats && (
        <div className="absolute right-4 top-4 z-20 flex items-center gap-1.5 rounded-full bg-black/25 px-3 py-1.5 text-xs text-white/90 backdrop-blur-md">
          <Database className="h-3.5 w-3.5" />
          {stats.files} files · {stats.chunks} chunks
        </div>
      )}

      {/* Conversation (only once there are messages) */}
      {!empty && (
        <div ref={scrollRef} className="rag-scroll w-full flex-1 overflow-y-auto">
          <div className="mx-auto flex w-full max-w-2xl flex-col gap-4 px-4 py-6">
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
          </div>
        </div>
      )}

      {/* Prompt box: centered when empty, pinned to bottom during a conversation */}
      <div
        className={cn(
          "z-10 w-full px-4",
          empty ? "flex flex-1 flex-col items-center justify-center" : "pb-6"
        )}
      >
        <div className="mx-auto w-full max-w-[600px]">
          {empty && (
            <div className="mb-6 text-center">
              <h1 className="text-3xl font-semibold text-white drop-shadow-sm">Ask your documents</h1>
              <p className="mt-2 text-sm text-white/80 drop-shadow-sm">
                Upload PDFs, Excel sheets or notes — then ask anything.
              </p>
            </div>
          )}

          <PromptInputBox
            onSend={(message) => handleAsk(message)}
            isLoading={isAsking}
            placeholder="Type your message here..."
          />

          {/* Subtle document-upload affordance */}
          <div className="mt-3 flex justify-center">
            <button
              onClick={() => docInputRef.current?.click()}
              disabled={isUploading}
              className="inline-flex items-center gap-2 rounded-full bg-black/20 px-4 py-1.5 text-xs text-white/90 backdrop-blur-md transition hover:bg-black/30 disabled:opacity-60"
            >
              {isUploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
              Add documents — or drop files anywhere (PDF · Excel · CSV · TXT · MD)
            </button>
          </div>
        </div>
      </div>

      <input
        ref={docInputRef}
        type="file"
        multiple
        accept={DOC_EXTENSIONS}
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) handleUpload(Array.from(e.target.files));
          e.target.value = "";
        }}
      />

      {/* Drag overlay */}
      {dragging && (
        <div className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-white/70 px-12 py-10">
            <Upload className="h-10 w-10 text-white" />
            <p className="text-lg font-medium text-white">Drop files to add them</p>
            <p className="text-sm text-white/70">PDF · Excel · CSV · TXT · MD</p>
          </div>
        </div>
      )}

      {/* Toasts */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={cn(
              "flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm shadow-lg backdrop-blur-md",
              t.kind === "success" && "border-green-400/40 bg-green-900/50 text-green-100",
              t.kind === "error" && "border-red-400/40 bg-red-900/50 text-red-100",
              t.kind === "info" && "border-white/15 bg-black/50 text-white"
            )}
          >
            {t.kind === "info" && <Loader2 className="h-4 w-4 animate-spin" />}
            {t.kind === "success" && <FileText className="h-4 w-4" />}
            {t.kind === "error" && <AlertCircle className="h-4 w-4" />}
            {t.text}
          </div>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-white/90 px-4 py-2.5 text-[#1F2023] shadow-md">
          {message.content}
        </div>
      </div>
    );
  }
  const pending = !message.content && !message.error;
  return (
    <div className="flex flex-col gap-2">
      <div
        className={cn(
          "max-w-[90%] whitespace-pre-wrap rounded-2xl rounded-bl-md border px-4 py-3 leading-relaxed shadow-lg backdrop-blur-md",
          message.error
            ? "border-red-400/40 bg-red-950/60 text-red-100"
            : "border-white/10 bg-[#1F2023]/85 text-gray-100"
        )}
      >
        {pending ? (
          <span className="flex items-center gap-2 text-gray-300">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">Searching documents &amp; generating…</span>
          </span>
        ) : (
          message.content
        )}
        {message.elapsed != null && (
          <span className="mt-2 block text-[11px] text-gray-400">answered in {message.elapsed.toFixed(1)}s</span>
        )}
      </div>
      {message.sources && message.sources.length > 0 && <Sources sources={message.sources} />}
    </div>
  );
}

function Sources({ sources }: { sources: Source[] }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="max-w-[90%]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1 text-xs text-white/80 transition hover:text-white"
      >
        <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
        {sources.length} source{sources.length > 1 ? "s" : ""}
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-2">
          {sources.map((s, i) => (
            <div
              key={i}
              className="rounded-lg border border-white/10 bg-[#161619]/85 px-3 py-2 text-xs backdrop-blur-md"
            >
              <div className="mb-1 flex items-center justify-between gap-2 text-gray-300">
                <span className="flex items-center gap-1.5 truncate">
                  <FileText className="h-3.5 w-3.5 shrink-0" />
                  <span className="truncate">{s.source}</span>
                </span>
                <span className="shrink-0 rounded-full bg-white/10 px-2 py-0.5 text-[10px] text-gray-200">
                  {(s.rerank_score * 100).toFixed(0)}%
                </span>
              </div>
              <p className="line-clamp-3 text-gray-400">{s.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


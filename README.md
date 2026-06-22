# RAG Pipeline (LanceDB + BGE-M3 + Reranker + Ollama)

Upgraded RAG pipeline aimed at much higher answer quality than a plain
vector-only setup. Four key changes vs. the typical `nomic-embed-text +
LanceDB + Qwen` setup:

1. **Cross-encoder reranking** with `BGE-reranker-v2-m3`
   — single biggest quality win.
2. **Hybrid search** (vector + BM25/FTS) fused with Reciprocal Rank Fusion
   — catches exact keyword matches that embeddings miss.
3. **`BGE-M3` embeddings** (1024 dim, multilingual)
   — far better than `nomic-embed-text` for Arabic / mixed content.
4. **Smarter chunking** (500/80, recursive, Arabic-aware separators).

The LLM (Qwen via Ollama) is untouched on purpose — that's the smallest
lever in a well-built RAG pipeline.

## Architecture

```
                Query
                  │
        ┌─────────▼─────────┐
        │   BGE-M3 embed    │
        └─────────┬─────────┘
                  │
   ┌──────────────┼──────────────┐
   ▼              ▼              │
Vector         BM25 (FTS)        │  Hybrid retrieval
search         search            │  (LanceDB)
   │              │              │
   └──────┬───────┘              │
          ▼                      │
     RRF fusion → top-20 ───────┘
          │
          ▼
  BGE-reranker-v2-m3 → top-5
          │
          ▼
    Qwen (Ollama) → answer
```

## المتطلبات المسبقة (Prerequisites)

- **Python 3.11** (تم الاختبار على 3.11.9).
- **Ollama** مثبّت وشغّال، ومعاه الموديلات المطلوبة:
  ```powershell
  ollama pull qwen2.5:7b          # الـ LLM المستخدم في config.py
  ollama list                     # للتأكد إنه موجود
  ```
- **مساحة قرص ~3.5 GB** لموديلات HuggingFace (BGE-M3 ~2.3GB + reranker ~600MB) — بتتحمّل أوتوماتيك أول تشغيل.

## التحميل والتثبيت (Setup)

### طريقة 1 — Windows / PowerShell (الإعداد الفعلي المُختبَر على CPU)

```powershell
# 1. إنشاء وتفعيل بيئة افتراضية
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. تثبيت المتطلبات الأساسية
pip install -r requirements.txt

# 3. متطلبات إضافية لازمة:
#    - pypdf لقراءة ملفات PDF
#    - transformers أقل من 5.x (الإصدارات 5.x بتكسر الـ reranker)
pip install pypdf "transformers>=4.44.2,<5.0"
```

> **ملاحظة CPU:** لو torch المثبّت نسخة CPU (من غير CUDA)، الـ [config.py](config.py)
> مظبوط بالفعل على `embedding_device="cpu"` و `embedding_use_fp16=False`.
> لو عندك GPU وثبّتت نسخة CUDA من torch، غيّرهم لـ `"cuda"` و `True`.

### طريقة 2 — Linux / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pypdf "transformers>=4.44.2,<5.0"
```

### (اختياري) تحميل موديلات BGE مسبقًا عشان أول تشغيل يبقى أسرع

```powershell
.\.venv\Scripts\python.exe -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
.\.venv\Scripts\python.exe -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3')"
```

## خطوات التشغيل (Run)

### 1) إدخال المستندات (Ingestion)

```powershell
# ملف PDF (مدعوم عبر pypdf)
.\.venv\Scripts\python.exe ingest.py "SQL.pdf"

# مجلد كامل من ملفات نصية
.\.venv\Scripts\python.exe ingest.py .\my_docs --ext .txt,.md
```
بيقسّم المستند لـ chunks، يعمل embedding، يخزّنهم في LanceDB، ويبني فهرس BM25.

### 2) السؤال (Query)

```powershell
# سؤال واحد
.\.venv\Scripts\python.exe query.py "What is a primary key in SQL?"

# وضع تفاعلي (أسرع — الموديلات بتتحمّل مرة واحدة)
.\.venv\Scripts\python.exe query.py -i
```

### 3) الاختبار من الكود (Test harness)

[test_rag.py](test_rag.py) بيحمّل الـ pipeline مرة واحدة ويشغّل قايمة أسئلة:

```powershell
# يجرب قايمة الأسئلة اللي جوه السكربت
.\.venv\Scripts\python.exe test_rag.py

# أو سؤال واحد من سطر الأوامر
.\.venv\Scripts\python.exe test_rag.py "What is a foreign key?"
```
لتجربة أسئلتك الخاصة، عدّل الـ `QUESTIONS` list في أول [test_rag.py](test_rag.py).

### الاستخدام مباشرة من الكود (Library)

```python
from rag import RAGPipeline

rag = RAGPipeline()

# إدخال
rag.ingest_text("Long document text here...", source="my_doc.txt")
rag.ingest_file("SQL.pdf")     # يدعم .pdf و .txt و .md
rag.build_indexes()            # تُستدعى مرة واحدة بعد انتهاء الإدخال

# سؤال
result = rag.query("What is the policy on X?")
print(result["answer"])
for src in result["sources"]:
    print(src["source"], src["rerank_score"])
```

## Hardware notes (RTX 4070 laptop, 8 GB VRAM)

Memory budget with defaults (Qwen 14B + BGE-M3 on GPU, reranker on CPU):

| Component            | VRAM      |
|----------------------|-----------|
| BGE-M3 (fp16)        | ~1.2 GB   |
| Qwen 2.5 14B (Q4_K_M)| ~9 GB     |
| Reranker (CPU)       | 0 GB VRAM |

14B at Q4 is borderline on 8 GB. If you see Ollama offloading to CPU and
generation gets slow, drop to `qwen2.5:7b` in `config.py`. Quality stays
high because the reranker already filtered the context heavily.

## Tuning knobs (in `config.py`)

| Setting              | Default | When to change                              |
|----------------------|---------|---------------------------------------------|
| `chunk_size`         | 500     | Bigger (800–1200) for tables / code blocks  |
| `chunk_overlap`      | 80      | Bigger if answers split across chunk borders|
| `initial_top_k`      | 20      | 30–50 if recall is the bottleneck           |
| `final_top_k`        | 5       | 3 for narrow questions, 8 for synthesis     |
| `vector_weight`      | 0.5     | ↑ for paraphrased/semantic queries          |
| `bm25_weight`        | 0.5     | ↑ for exact keywords, IDs, acronyms         |
| `temperature`        | 0.1     | Keep low for factual QA                     |

## Quality debugging

When an answer is wrong, check in this order:

1. **Are the right chunks in `initial_top_k`?** If no, raise it or shift
   `bm25_weight` up (most "lost" matches are keyword-shaped).
2. **Are they in `final_top_k` after reranking?** If they're at rank 6–10,
   bump `final_top_k`. If they're not in the rerank input at all, problem
   is upstream.
3. **Are they in the final context but the answer still wrong?** Now it's
   an LLM issue — try Qwen 14B if you're on 7B, or tighten the system
   prompt in `llm.py`.

This ordering is on purpose: 80% of RAG quality issues are retrieval,
not generation.

## استكشاف الأخطاء (Troubleshooting)

| المشكلة | السبب | الحل |
|---------|-------|------|
| `XLMRobertaTokenizer has no attribute prepare_for_model` | `transformers` 5.x غير متوافق مع الـ reranker | `pip install "transformers>=4.44.2,<5.0"` |
| `UnicodeEncodeError: 'charmap' codec...` عند طباعة المصادر | الـ console على ويندوز بيستخدم cp1252 | تم حلّه في [query.py](query.py) و [test_rag.py](test_rag.py) بإجبار UTF-8 |
| الإدخال بطيء جدًا / يفشل على PDF | `pypdf` غير مثبّت | `pip install pypdf` |
| `cuda` errors أو بطء شديد | torch نسخة CPU بس الـ config على `cuda` | خلّي `embedding_device="cpu"` في [config.py](config.py) |
| أول تشغيل بطيء (دقايق) | تحميل موديلات BGE من HuggingFace | طبيعي — مرة واحدة بس، بعدها بتتكاش |

## File layout

```
rag/
├── config.py         # All knobs in one place
├── chunking.py       # Recursive chunker with Arabic/English separators
├── embeddings.py     # BGE-M3 wrapper
├── reranker.py       # BGE-reranker-v2-m3 wrapper
├── vector_store.py   # LanceDB + hybrid search + RRF fusion
├── llm.py            # Ollama client + RAG prompt template
├── rag.py            # Pipeline that wires everything together (PDF/txt/md ingest)
├── ingest.py         # CLI: ingest files
├── query.py          # CLI: ask questions
├── test_rag.py       # Test harness: load once, run a batch of questions
├── requirements.txt
└── README.md
```

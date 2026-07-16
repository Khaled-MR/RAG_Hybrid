# RAG Pipeline (Qdrant + BGE-M3 + Reranker + Ollama + Web UI)

نظام RAG كامل (backend + واجهة ويب) مظبوط لإجابات **دقيقة وسريعة** على مستنداتك
(PDF / Excel / CSV / TXT / MD) — عربي وإنجليزي — مع GPU و streaming.

أهم نقاط القوة مقارنة بـ RAG عادي (vector-only):

1. **Cross-encoder reranking** بـ `BGE-reranker-v2-m3` — أكبر مكسب في الجودة.
2. **بحث هجين** في Qdrant (dense + sparse من BGE-M3) مدموج بـ RRF — بيمسك
   الكلمات المفتاحية الدقيقة (أرقام مواد، أكواد، اختصارات) اللي الـ embeddings
   بتفوّتها.
3. **`BGE-M3` embeddings** (1024-dim، multilingual) — ممتاز للعربي.
4. **تقطيع واعٍ بالبنية** — كل «المادة N» / عنوان / قسم بيبقى chunk مستقل كامل،
   فالموديل يقدر يقتبس النص حرفياً ويربط بين المواد.
5. **برومبت تأريض صارم + temperature=0** — أقل هلوسة.
6. **GPU + streaming + keep_alive** — أول كلمة في ~0.5 ثانية، إجابة كاملة في ~3 ثواني.

## Architecture

```
                Query
                  │
        ┌─────────▼─────────┐
        │ BGE-M3 dense+sparse│  (GPU)
        └─────────┬─────────┘
                  │
   ┌──────────────┼──────────────┐
   ▼              ▼              │
Dense          Sparse            │  Hybrid retrieval (Qdrant, embedded)
search         search            │
   │              │              │
   └──────┬───────┘              │
          ▼                      │
     RRF fusion → top-40 ───────┘
          │
          ▼
  BGE-reranker-v2-m3 → top-6   (GPU)
          │
          ▼
   llama3.1:8b (Ollama) → streamed answer + sources
```

## المتطلبات (Prerequisites)

- **Python 3.11** و **Node.js 18+** (للواجهة).
- **Ollama** شغّال + الموديل:
  ```powershell
  ollama pull llama3.1        # الـ LLM الافتراضي في config.py
  ```
- **مساحة ~3.5 GB** لموديلات BGE (بتتحمّل أوتوماتيك أول تشغيل).

## التثبيت (Setup)

```powershell
# 1) بيئة افتراضية
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) متطلبات الـ backend
pip install -r requirements.txt
pip install pypdf openpyxl "transformers>=4.44.2,<5.0"

# 3) (للـ GPU) torch بنسخة CUDA — RTX 50-series محتاجة cu128
.\.venv\Scripts\python.exe -m pip install --no-deps `
  --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"

# تأكيد إن الـ GPU شُغّال:
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # -> True
```

> الـ device في [config.py](backend/config.py) = `"auto"`: بيستخدم الـ GPU للـ
> embedder/reranker لو الكارت **≥10GB**، وإلا بيحطّهم على CPU عشان يسيب الـ GPU
> بالكامل للـ LLM (الأهم على كروت 8GB زي RTX 4070). من غير CUDA بيرجع CPU تلقائياً.

---

## 🚀 التشغيل (Run)

### الواجهة الكاملة (backend + frontend)

```powershell
# Terminal 1 — الـ backend (API على بورت 8000)
cd backend
..\.venv\Scripts\python.exe -m uvicorn api:app --port 8000

# Terminal 2 — الـ frontend (بورت 5173)
cd frontend
npm install        # أول مرة بس
npm run dev
```

افتح **http://localhost:5173**:
- **اكتب سؤالك** → الإجابة بتظهر **streaming** (كلمة كلمة)، وتحتها زرار
  **"sources"** يوريك المصادر مع نسبة التطابق.
- **زرار "Add documents"** أو اسحب أي ملف على الصفحة → رفع + ingest فوري.

### إدخال آلاف الملفات (CLI)

```powershell
# 1) حط ملفاتك في  backend/data/    2) شغّل:
cd backend
..\.venv\Scripts\python.exe ingest.py
```

بيمشي على `data/` كله (recursively)، يدعم `.pdf .xlsx .xls .csv .txt .md`،
يقطّع، يعمل embedding على الـ GPU، يخزّن في LanceDB، ويبني فهارس BM25 + ANN.

**مظبوط لـ 43 ألف ملف:**
- **قابل للاستئناف:** manifest للملفات الخلصانة — لو وقف، شغّله تاني يكمّل.
- **عزل الأخطاء:** ملف باظ مايوقفش الباقي (بيتسجّل في `ingest_errors.log`).
- أوامر إضافية:
  ```powershell
  ..\.venv\Scripts\python.exe ingest.py "C:\path\to\file.pdf"   # ملف واحد
  ..\.venv\Scripts\python.exe ingest.py --ext .pdf,.xlsx        # أنواع محددة
  ..\.venv\Scripts\python.exe ingest.py --reset-manifest        # إعادة إدخال الكل
  ```

> ⚠️ بعد أي تغيير في إعدادات التقطيع، أعِد الإدخال بـ `--reset-manifest`
> عشان الملفات تتقطّع بالطريقة الجديدة.

### من سطر الأوامر / كود

```powershell
cd backend
..\.venv\Scripts\python.exe query.py "ما هو الـ primary key؟"   # سؤال واحد
..\.venv\Scripts\python.exe query.py -i                          # وضع تفاعلي
..\.venv\Scripts\python.exe benchmark.py                         # قياس الأداء
```

```python
from rag import RAGPipeline
rag = RAGPipeline()
rag.ingest_file("data/contract.pdf"); rag.build_indexes()
print(rag.query("اشرح المادة 17 واربطها بالمواد الأخرى")["answer"])
```

---

## مفاتيح الضبط (في [config.py](backend/config.py))

| Setting | Default | امتى تغيّره |
|---------|---------|------------|
| `llm_model` | `llama3.1:latest` | موديل تاني من Ollama |
| `chunk_size` / `chunk_overlap` | 900 / 150 | أكبر للجداول/الفقرات الطويلة |
| `initial_top_k` / `final_top_k` | 40 / 6 | ارفعهم لو الاسترجاع بيفوّت |
| `vector_weight` / `bm25_weight` | 0.5 / 0.5 | ↑ bm25 للكلمات/الأرقام الدقيقة |
| `temperature` | 0.0 | خلّيه 0 للأسئلة الواقعية |
| `max_tokens` | 700 | أقل = أسرع |
| `num_ctx` | 4096 | ارفعه لو زوّدت `final_top_k`/`chunk_size` كتير |
| `llm_keep_alive` | `30m` | `-1` يخلّي الموديل محمّل دايماً |
| `embedding_device` / `reranker_device` | `auto` | `cuda` / `cpu` بالإجبار |

## تشخيص جودة الإجابة (الترتيب مقصود — 80% من المشاكل في الاسترجاع)

1. هل الـ chunk الصح ضمن `initial_top_k`؟ لو لأ → ارفعه أو زوّد `bm25_weight`.
2. هل فضل بعد الـ rerank ضمن `final_top_k`؟ لو في رتبة 7–10 → ارفع `final_top_k`.
3. الـ chunk صح والإجابة غلط؟ دي مشكلة LLM → جرّب موديل أكبر أو شدّ البرومبت في
   [llm.py](backend/llm.py).

## Troubleshooting

| المشكلة | الحل |
|---------|------|
| `torch.cuda.is_available()` → False | ثبّت torch cu128 (فوق)؛ غير كده بيشتغل على CPU |
| الرد بطيء جداً على 8GB | الموديل بيتسرّب للـ CPU — استخدم موديل ≤8B وسيب `device="auto"` |
| `transformers` 5.x بيكسر الـ reranker | `pip install "transformers>=4.44.2,<5.0"` |
| `/api/ask/stream` → Not Found | في سيرفر قديم شغّال — اقفله وشغّل واحد |
| `Storage folder ... already accessed by another instance` | Qdrant المحلي بيتفتح من عملية واحدة بس — اقفل الـ backend قبل ما تشغّل `ingest.py`، أو ارفع الملفات من الواجهة بدل الـ CLI |
| `meta tensor` عند التحميل | اتنين backend بيحمّلوا الموديلات مع بعض — شغّل **واحد بس** |
| أول تشغيل بطيء (دقايق) | تحميل موديلات BGE — مرة واحدة بس |

## File layout

```
rag/
├── backend/                 # كل كود الـ RAG (Python) + الـ API
│   ├── config.py            # كل الإعدادات
│   ├── chunking.py          # تقطيع واعٍ بالبنية (مواد/عناوين)
│   ├── embeddings.py        # BGE-M3
│   ├── reranker.py          # BGE-reranker-v2-m3
│   ├── qdrant_store.py      # Qdrant (embedded): dense+sparse hybrid + RRF
│   ├── vector_store.py      # legacy LanceDB store (unused; kept for reference)
│   ├── llm.py               # Ollama client + برومبت + streaming
│   ├── rag.py               # الـ pipeline (PDF/xlsx/csv/txt/md)
│   ├── ingest.py            # CLI إدخال بالجملة (resumable)
│   ├── query.py             # CLI أسئلة
│   ├── benchmark.py         # قياس السرعة/الأداء
│   ├── api.py               # FastAPI (يشمل /api/ask/stream)
│   ├── data/                # ← حط ملفاتك هنا
│   └── qdrant_db/           # قاعدة Qdrant المحلية (أوتوماتيك)
├── frontend/                # React + Vite + Tailwind + shadcn
│   └── src/{App.tsx, lib/api.ts, components/ui/ai-prompt-box.tsx}
├── requirements.txt
└── README.md
```

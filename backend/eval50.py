# -*- coding: utf-8 -*-
"""
50-question evaluation of the RAG system.

Measures, per question:
  * objective metrics: latency, TTFT, generation time, tokens, throughput,
    citation present, answer-language match, retrieval strength (rerank),
    extracted confidence + human-review flag, refusal detection.
  * LLM-as-judge (1-5): faithfulness (anti-hallucination), relevance,
    completeness, human-likeness/convincing; plus a correct (bool) verdict.

Out-of-corpus questions test whether the system correctly REFUSES instead of
hallucinating.

Writes a Markdown report. Run inside the backend container:
    docker compose exec -T backend python eval50.py
"""
import json
import re
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from rag import RAGPipeline
from config import RAGConfig

# Overridable by other eval scripts (e.g. eval_legal.py) before calling main().
OUTPUT_FILE = "EVAL_REPORT.md"
REPORT_TITLE = "RAG Evaluation Report — 50 Questions"

# (question, category, in_corpus)
QUESTIONS = [
    # ---- RAG ----
    ("What is Retrieval-Augmented Generation?", "RAG", True),
    ("What are the main steps of a RAG pipeline?", "RAG", True),
    ("Why is reranking important in RAG?", "RAG", True),
    ("ما هو الـ RAG وكيف يعمل؟", "RAG", True),
    ("What is a vector database used for?", "RAG", True),
    # ---- LLMs / transformers ----
    ("What is a transformer architecture?", "LLM", True),
    ("Explain the attention mechanism.", "LLM", True),
    ("What is a large language model?", "LLM", True),
    ("What is a context window in an LLM?", "LLM", True),
    ("What is tokenization in NLP?", "LLM", True),
    ("What is a word embedding?", "LLM", True),
    ("What is the difference between fine-tuning and prompt engineering?", "LLM", True),
    ("ما هي آلية الانتباه attention؟", "LLM", True),
    ("ما هو نموذج اللغة الكبير LLM؟", "LLM", True),
    ("What is the difference between an encoder and a decoder?", "LLM", True),
    # ---- Docker ----
    ("What is a Docker container?", "Docker", True),
    ("What is the difference between a Docker image and a container?", "Docker", True),
    ("What is a Dockerfile?", "Docker", True),
    ("ما هو الـ Docker container؟", "Docker", True),
    ("How does Docker isolate applications?", "Docker", True),
    # ---- SQL ----
    ("What is a primary key in SQL?", "SQL", True),
    ("What is a foreign key?", "SQL", True),
    ("What is the difference between DELETE and TRUNCATE?", "SQL", True),
    ("What does the JOIN keyword do?", "SQL", True),
    ("How do you create a table with a PRIMARY KEY in MySQL?", "SQL", True),
    ("What is an SQL index?", "SQL", True),
    ("ما هو المفتاح الأساسي في SQL؟", "SQL", True),
    # ---- AI agents ----
    ("How do you build a reliable AI agent?", "Agents", True),
    ("How do you evaluate an AI agent?", "Agents", True),
    ("What are common ways to build AI agents with confidence?", "Agents", True),
    ("ما هي طرق بناء وكلاء الذكاء الاصطناعي؟", "Agents", True),
    # ---- NLP / data science ----
    ("What is named entity recognition?", "NLP", True),
    ("What is transfer learning in NLP?", "NLP", True),
    ("What is semantic search?", "NLP", True),
    ("What is multimodality in language models?", "NLP", True),
    ("What is the Querying Transformer (Q-Former)?", "NLP", True),
    ("What is model quantization?", "NLP", True),
    ("ما هو البحث الدلالي؟", "NLP", True),
    ("What is data leakage in machine learning?", "DataScience", True),
    ("What is cross-validation?", "DataScience", True),
    ("What is feature engineering?", "DataScience", True),
    # ---- Out-of-corpus (should REFUSE, not hallucinate) ----
    ("ما هي عقوبة مخالفة قانون حماية البيانات المصري رقم 151؟", "OOC", False),
    ("ما المدة الزمنية المحددة في المادة 44 لتسجيل نشاط المعالجة؟", "OOC", False),
    ("What is the capital of Australia?", "OOC", False),
    ("ما هي جرعة دواء الباراسيتامول للأطفال؟", "OOC", False),
    ("Who won the 2026 FIFA World Cup?", "OOC", False),
    ("ما هو سعر البيتكوين اليوم؟", "OOC", False),
    ("What are the side effects of ibuprofen?", "OOC", False),
    ("ما هي شروط الحصول على تأشيرة شنغن؟", "OOC", False),
    ("How do I file taxes in Egypt?", "OOC", False),
    ("What is the boiling point of mercury?", "OOC", False),
]

JUDGE_MODEL = None  # set below to config.llm_model

JUDGE_PROMPT = (
    "You are a STRICT evaluator of a retrieval-augmented answer. You are given "
    "the QUESTION, the SOURCES that were retrieved, and the ANSWER.\n"
    "Score each 1-5 (5=best):\n"
    "- faithfulness: are ALL factual claims in the answer supported by the "
    "SOURCES? (5 = fully grounded, 1 = hallucinated / unsupported)\n"
    "- relevance: does the answer actually address the question?\n"
    "- completeness: is it complete given what the sources contain?\n"
    "- humanlike: is it fluent, natural and convincing (reads like a "
    "knowledgeable human, not robotic)?\n"
    "Also set correct=true if the answer is factually correct given the sources, "
    "OR if it correctly states the information is not available (when the sources "
    "don't contain it). correct=false if it makes unsupported/incorrect claims.\n"
    'Output ONLY compact JSON: {"faithfulness":N,"relevance":N,'
    '"completeness":N,"humanlike":N,"correct":true}'
)

AR_RANGE = lambda c: "؀" <= c <= "ۿ"


def q_lang(t):
    return "ar" if sum(1 for c in t if AR_RANGE(c)) >= 2 else "en"


def is_refusal(ans):
    pats = ["لا يوجد", "لا تتوفر", "غير متوفر", "لا أستطيع", "لا يمكنني",
            "not available", "no information", "not in the", "cannot find",
            "don't have", "does not contain", "لا توجد"]
    a = ans.lower()
    return any(p.lower() in a for p in pats)


def extract_confidence(ans):
    m = re.search(r"(?:درجة الثقة|confidence)\s*[:：]?\s*(عالية|متوسطة|منخفضة|high|medium|low)",
                  ans, re.IGNORECASE)
    return (m.group(1) if m else "").lower()


def needs_review(ans):
    m = re.search(r"(?:يحتاج مراجعة بشرية|human review)\s*[:：]?\s*(نعم|لا|yes|no)",
                  ans, re.IGNORECASE)
    return (m.group(1) if m else "").lower()


def has_citation(ans):
    return bool(re.search(r"\[\s*\d+\s*\]|\[المصدر|\[Source", ans, re.IGNORECASE))


def measure(rag, q):
    t0 = time.time()
    search_q = rag.rewrite_query(q)
    extra = [q] if rag.config.enable_multi_query else None
    retrieved = rag.retrieve(search_q, extra_queries=extra)
    t_ret = time.time() - t0
    contexts = rag.format_contexts(rag._expand_with_neighbors(retrieved))

    g0 = time.time(); ttft = None; out = ""
    stream = rag.llm.client.chat(
        model=rag.llm.model, messages=rag.llm._build_messages(q, contexts, None),
        keep_alive=rag.llm.keep_alive, stream=True,
        options={"temperature": 0.0, "num_predict": rag.config.max_tokens,
                 "num_ctx": rag.config.num_ctx},
    )
    last = None
    for part in stream:
        c = part.get("message", {}).get("content", "")
        if c and ttft is None:
            ttft = time.time() - g0
        out += c
        last = part
    gen_s = time.time() - g0
    ec = (last or {}).get("eval_count", 0)
    ed = (last or {}).get("eval_duration", 1) / 1e9
    top_scores = [s.get("rerank_score", 0.0) for s in retrieved[:3]]
    return {
        "answer": out,
        "retrieve_s": t_ret, "ttft_s": ttft or gen_s, "gen_s": gen_s,
        "total_s": t_ret + gen_s, "out_tok": ec,
        "gen_tok_s": ec / ed if ed else 0.0,
        "sources": [s.get("text", "")[:600] for s in retrieved[:5]],
        "avg_rerank": statistics.mean(top_scores) if top_scores else 0.0,
    }


def judge(rag, q, answer, sources):
    src = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources)) or "(none)"
    try:
        r = rag.llm.client.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user", "content":
                       f"QUESTION:\n{q}\n\nSOURCES:\n{src}\n\nANSWER:\n{answer}"}],
            keep_alive="30m",
            options={"temperature": 0.0, "num_predict": 120, "num_ctx": 8192},
        )
        txt = r["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        d = json.loads(m.group(0))
        return {
            "faithfulness": float(d.get("faithfulness", 0)),
            "relevance": float(d.get("relevance", 0)),
            "completeness": float(d.get("completeness", 0)),
            "humanlike": float(d.get("humanlike", 0)),
            "correct": bool(d.get("correct", False)),
        }
    except Exception as e:
        return {"faithfulness": 0, "relevance": 0, "completeness": 0,
                "humanlike": 0, "correct": False, "err": str(e)[:60]}


def main():
    global JUDGE_MODEL
    cfg = RAGConfig()
    JUDGE_MODEL = cfg.llm_model
    print("Loading pipeline...", file=sys.stderr)
    rag = RAGPipeline(cfg)
    list(rag.llm.generate_stream("warmup", ["warmup"], max_tokens=4))

    rows = []
    for i, (q, cat, in_corpus) in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {cat}: {q[:45]}", file=sys.stderr)
        m = measure(rag, q)
        j = judge(rag, q, m["answer"], m["sources"])
        rows.append({
            "q": q, "cat": cat, "in_corpus": in_corpus,
            "lang": q_lang(q), "ans_lang": q_lang(m["answer"]),
            "refusal": is_refusal(m["answer"]),
            "citation": has_citation(m["answer"]),
            "confidence": extract_confidence(m["answer"]),
            "review": needs_review(m["answer"]),
            **{k: m[k] for k in ("retrieve_s", "ttft_s", "gen_s", "total_s",
                                 "out_tok", "gen_tok_s", "avg_rerank")},
            **{f"j_{k}": v for k, v in j.items()},
            "answer": m["answer"],
        })

    write_report(cfg, rows)


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.mean(vals) if vals else 0.0


def pct(rows, cond):
    return 100.0 * sum(1 for r in rows if cond(r)) / len(rows) if rows else 0.0


def write_report(cfg, rows):
    ic = [r for r in rows if r["in_corpus"]]
    ooc = [r for r in rows if not r["in_corpus"]]
    o = []; w = o.append

    w(f"# {REPORT_TITLE}\n")
    w(f"- System: `{cfg.llm_model}` · hybrid (dense+sparse) + reranker + "
      f"query-rewrite + multi-query + neighbor-expansion")
    w(f"- {len(rows)} questions ({len(ic)} in-corpus, {len(ooc)} out-of-corpus refusal tests)")
    w(f"- Judge: LLM-as-judge with `{cfg.llm_model}` (self-judge — treat quality "
      f"scores as directional; objective metrics are exact)\n")

    w("## 1) Speed / performance (objective)\n")
    w("| Metric | Mean |")
    w("|--------|-----:|")
    w(f"| Total latency (s) | {mean(rows,'total_s'):.2f} |")
    w(f"| Time to first token (s) | {mean(rows,'ttft_s'):.2f} |")
    w(f"| Retrieval time (s) | {mean(rows,'retrieve_s'):.2f} |")
    w(f"| Generation time (s) | {mean(rows,'gen_s'):.2f} |")
    w(f"| Output tokens | {mean(rows,'out_tok'):.0f} |")
    w(f"| Generation throughput (tok/s) | {mean(rows,'gen_tok_s'):.1f} |")
    tot = sorted(r["total_s"] for r in rows)
    w(f"\n- p50 total: {tot[len(tot)//2]:.2f}s · p95 total: {tot[int(len(tot)*0.95)-1]:.2f}s\n")

    w("## 2) Quality (LLM-judge, 1-5) — in-corpus\n")
    w("| Dimension | Mean |")
    w("|-----------|-----:|")
    for k, label in [("j_faithfulness", "Faithfulness (anti-hallucination)"),
                     ("j_relevance", "Relevance"),
                     ("j_completeness", "Completeness"),
                     ("j_humanlike", "Human-like / convincing")]:
        w(f"| {label} | {mean(ic,k):.2f} / 5 |")
    w(f"\n- **Correct (in-corpus):** {pct(ic, lambda r: r['j_correct']):.0f}%")
    w(f"- **Has citation:** {pct(ic, lambda r: r['citation']):.0f}%")
    w(f"- **Answer language matches question:** {pct(rows, lambda r: r['lang']==r['ans_lang']):.0f}%\n")

    w("## 3) Hallucination / refusal (out-of-corpus)\n")
    w("These questions are NOT in the documents — the system SHOULD refuse.\n")
    w(f"- **Correctly refused:** {pct(ooc, lambda r: r['refusal']):.0f}%  "
      f"(higher = better; means it didn't hallucinate)")
    w(f"- **Judge marked 'correct' (i.e. correctly refused):** {pct(ooc, lambda r: r['j_correct']):.0f}%")
    w(f"- **In-corpus faithfulness ≥ 4/5:** {pct(ic, lambda r: r['j_faithfulness']>=4):.0f}%\n")

    w("## 4) Grounding metadata (structured output)\n")
    conf = {}
    for r in rows:
        conf[r["confidence"] or "—"] = conf.get(r["confidence"] or "—", 0) + 1
    w(f"- Confidence labels emitted: {dict(conf)}")
    w(f"- Human-review flag present: {pct(rows, lambda r: bool(r['review'])):.0f}% of answers\n")

    w("## 5) By category\n")
    w("| Category | n | avg total (s) | faithful | relevance | correct% |")
    w("|----------|--:|----:|----:|----:|----:|")
    cats = {}
    for r in rows:
        cats.setdefault(r["cat"], []).append(r)
    for cat, rs in cats.items():
        w(f"| {cat} | {len(rs)} | {mean(rs,'total_s'):.2f} | "
          f"{mean(rs,'j_faithfulness'):.1f} | {mean(rs,'j_relevance'):.1f} | "
          f"{pct(rs, lambda r: r['j_correct']):.0f}% |")

    w("\n## 6) Per-question detail\n")
    w("| # | Cat | Question | tot(s) | rerank | cite | conf | faith | rel | comp | human | correct |")
    w("|---|-----|----------|----:|----:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    for i, r in enumerate(rows, 1):
        q = (r["q"][:36] + "…") if len(r["q"]) > 37 else r["q"]
        w(f"| {i} | {r['cat']} | {q} | {r['total_s']:.1f} | {r['avg_rerank']:.2f} "
          f"| {'✓' if r['citation'] else '·'} | {r['confidence'] or '—'} "
          f"| {r['j_faithfulness']:.0f} | {r['j_relevance']:.0f} | {r['j_completeness']:.0f} "
          f"| {r['j_humanlike']:.0f} | {'✓' if r['j_correct'] else '✗'} |")

    # a few full sample answers
    w("\n## 7) Sample answers\n")
    for r in (ic[0], ic[3], ooc[0]):
        w(f"**Q ({r['cat']}):** {r['q']}\n")
        w(f"```\n{r['answer'][:900]}\n```\n")

    text = "\n".join(o)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n[saved to backend/{OUTPUT_FILE}]", file=sys.stderr)


if __name__ == "__main__":
    main()

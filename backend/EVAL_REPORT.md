# RAG Evaluation Report — 50 Questions

- System: `qwen2.5:7b` · hybrid (dense+sparse) + reranker + query-rewrite + multi-query + neighbor-expansion
- 51 questions (41 in-corpus, 10 out-of-corpus refusal tests)
- Judge: LLM-as-judge with `qwen2.5:7b` (self-judge — treat quality scores as directional; objective metrics are exact)

## 1) Speed / performance (objective)

| Metric | Mean |
|--------|-----:|
| Total latency (s) | 10.15 |
| Time to first token (s) | 3.60 |
| Retrieval time (s) | 3.64 |
| Generation time (s) | 6.51 |
| Output tokens | 231 |
| Generation throughput (tok/s) | 69.2 |

- p50 total: 9.25s · p95 total: 15.91s

## 2) Quality (LLM-judge, 1-5) — in-corpus

| Dimension | Mean |
|-----------|-----:|
| Faithfulness (anti-hallucination) | 3.15 / 5 |
| Relevance | 4.07 / 5 |
| Completeness | 3.29 / 5 |
| Human-like / convincing | 3.32 / 5 |

- **Correct (in-corpus):** 66%
- **Has citation:** 59%
- **Answer language matches question:** 24%

## 3) Hallucination / refusal (out-of-corpus)

These questions are NOT in the documents — the system SHOULD refuse.

- **Correctly refused:** 100%  (higher = better; means it didn't hallucinate)
- **Judge marked 'correct' (i.e. correctly refused):** 10%
- **In-corpus faithfulness ≥ 4/5:** 34%

## 4) Grounding metadata (structured output)

- Confidence labels emitted: {'عالية': 18, 'منخفضة': 30, '—': 3}
- Human-review flag present: 96% of answers

## 5) By category

| Category | n | avg total (s) | faithful | relevance | correct% |
|----------|--:|----:|----:|----:|----:|
| RAG | 5 | 10.60 | 3.4 | 4.2 | 60% |
| LLM | 10 | 9.12 | 3.3 | 4.1 | 70% |
| Docker | 5 | 8.81 | 3.4 | 4.2 | 80% |
| SQL | 7 | 10.26 | 3.6 | 4.4 | 100% |
| Agents | 4 | 15.17 | 3.2 | 4.2 | 50% |
| NLP | 7 | 8.90 | 2.6 | 3.7 | 43% |
| DataScience | 3 | 8.87 | 2.0 | 3.3 | 33% |
| OOC | 10 | 10.81 | 1.2 | 2.3 | 10% |

## 6) Per-question detail

| # | Cat | Question | tot(s) | rerank | cite | conf | faith | rel | comp | human | correct |
|---|-----|----------|----:|----:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | RAG | What is Retrieval-Augmented Generati… | 15.5 | 0.88 | ✓ | عالية | 5 | 5 | 5 | 4 | ✓ |
| 2 | RAG | What are the main steps of a RAG pip… | 10.0 | 0.56 | ✓ | عالية | 5 | 5 | 5 | 4 | ✓ |
| 3 | RAG | Why is reranking important in RAG? | 8.9 | 0.37 | · | منخفضة | 2 | 4 | 3 | 3 | ✗ |
| 4 | RAG | ما هو الـ RAG وكيف يعمل؟ | 9.2 | 0.59 | · | منخفضة | 3 | 4 | 3 | 3 | ✓ |
| 5 | RAG | What is a vector database used for? | 9.4 | 0.09 | · | منخفضة | 2 | 3 | 3 | 2 | ✗ |
| 6 | LLM | What is a transformer architecture? | 9.4 | 0.15 | · | — | 2 | 3 | 3 | 4 | ✗ |
| 7 | LLM | Explain the attention mechanism. | 9.4 | 0.19 | · | منخفضة | 2 | 3 | 3 | 4 | ✓ |
| 8 | LLM | What is a large language model? | 2.8 | 0.53 | · | منخفضة | 4 | 5 | 5 | 3 | ✓ |
| 9 | LLM | What is a context window in an LLM? | 8.6 | 0.44 | ✓ | عالية | 5 | 5 | 5 | 4 | ✓ |
| 10 | LLM | What is tokenization in NLP? | 9.4 | 0.97 | · | منخفضة | 3 | 4 | 3 | 2 | ✓ |
| 11 | LLM | What is a word embedding? | 9.1 | 0.68 | ✓ | منخفضة | 3 | 4 | 3 | 4 | ✓ |
| 12 | LLM | What is the difference between fine-… | 14.1 | 0.38 | ✓ | عالية | 3 | 4 | 3 | 4 | ✗ |
| 13 | LLM | ما هي آلية الانتباه attention؟ | 9.6 | 0.00 | · | منخفضة | 2 | 3 | 3 | 4 | ✗ |
| 14 | LLM | ما هو نموذج اللغة الكبير LLM؟ | 9.2 | 0.55 | · | منخفضة | 4 | 5 | 4 | 3 | ✓ |
| 15 | LLM | What is the difference between an en… | 9.7 | 0.59 | ✓ | عالية | 5 | 5 | 5 | 4 | ✓ |
| 16 | Docker | What is a Docker container? | 8.7 | 0.26 | ✓ | منخفضة | 2 | 3 | 2 | 3 | ✗ |
| 17 | Docker | What is the difference between a Doc… | 9.1 | 0.49 | · | منخفضة | 3 | 4 | 2 | 3 | ✓ |
| 18 | Docker | What is a Dockerfile? | 7.9 | 0.12 | ✓ | منخفضة | 3 | 4 | 2 | 3 | ✓ |
| 19 | Docker | ما هو الـ Docker container؟ | 9.1 | 0.88 | ✓ | عالية | 4 | 5 | 3 | 5 | ✓ |
| 20 | Docker | How does Docker isolate applications? | 9.4 | 0.15 | ✓ | عالية | 5 | 5 | 5 | 5 | ✓ |
| 21 | SQL | What is a primary key in SQL? | 8.0 | 0.36 | ✓ | عالية | 3 | 4 | 3 | 2 | ✓ |
| 22 | SQL | What is a foreign key? | 8.8 | 0.48 | ✓ | عالية | 3 | 4 | 2 | 3 | ✓ |
| 23 | SQL | What is the difference between DELET… | 11.1 | 0.12 | ✓ | — | 4 | 5 | 4 | 5 | ✓ |
| 24 | SQL | What does the JOIN keyword do? | 8.0 | 0.65 | ✓ | عالية | 3 | 4 | 2 | 3 | ✓ |
| 25 | SQL | How do you create a table with a PRI… | 17.8 | 0.85 | ✓ | عالية | 4 | 5 | 4 | 5 | ✓ |
| 26 | SQL | What is an SQL index? | 9.1 | 0.18 | ✓ | منخفضة | 3 | 4 | 3 | 2 | ✓ |
| 27 | SQL | ما هو المفتاح الأساسي في SQL؟ | 9.0 | 0.09 | ✓ | عالية | 5 | 5 | 5 | 5 | ✓ |
| 28 | Agents | How do you build a reliable AI agent? | 13.5 | 0.67 | ✓ | عالية | 4 | 5 | 4 | 5 | ✓ |
| 29 | Agents | How do you evaluate an AI agent? | 15.9 | 0.69 | ✓ | عالية | 3 | 4 | 3 | 2 | ✓ |
| 30 | Agents | What are common ways to build AI age… | 14.3 | 0.23 | ✓ | عالية | 3 | 4 | 2 | 3 | ✗ |
| 31 | Agents | ما هي طرق بناء وكلاء الذكاء الاصطناع… | 16.9 | 0.20 | ✓ | عالية | 3 | 4 | 3 | 4 | ✗ |
| 32 | NLP | What is named entity recognition? | 8.4 | 0.89 | · | منخفضة | 2 | 3 | 3 | 2 | ✗ |
| 33 | NLP | What is transfer learning in NLP? | 8.2 | 0.95 | · | منخفضة | 2 | 3 | 2 | 2 | ✗ |
| 34 | NLP | What is semantic search? | 8.4 | 0.17 | · | منخفضة | 2 | 3 | 2 | 3 | ✗ |
| 35 | NLP | What is multimodality in language mo… | 8.0 | 0.61 | · | منخفضة | 2 | 4 | 3 | 2 | ✗ |
| 36 | NLP | What is the Querying Transformer (Q-… | 10.5 | 0.54 | ✓ | عالية | 4 | 5 | 4 | 3 | ✓ |
| 37 | NLP | What is model quantization? | 8.1 | 0.09 | · | منخفضة | 2 | 3 | 3 | 2 | ✓ |
| 38 | NLP | ما هو البحث الدلالي؟ | 10.7 | 0.00 | ✓ | عالية | 4 | 5 | 4 | 5 | ✓ |
| 39 | DataScience | What is data leakage in machine lear… | 8.9 | 0.27 | · | منخفضة | 2 | 4 | 3 | 3 | ✗ |
| 40 | DataScience | What is cross-validation? | 9.9 | 0.03 | ✓ | منخفضة | 3 | 4 | 3 | 2 | ✓ |
| 41 | DataScience | What is feature engineering? | 7.8 | 0.33 | · | منخفضة | 1 | 2 | 3 | 2 | ✗ |
| 42 | OOC | ما هي عقوبة مخالفة قانون حماية البيا… | 10.7 | 0.00 | · | منخفضة | 2 | 3 | 3 | 3 | ✗ |
| 43 | OOC | ما المدة الزمنية المحددة في المادة 4… | 9.2 | 0.01 | ✓ | منخفضة | 2 | 1 | 3 | 3 | ✗ |
| 44 | OOC | What is the capital of Australia? | 8.1 | 0.15 | · | منخفضة | 1 | 2 | 3 | 3 | ✗ |
| 45 | OOC | ما هي جرعة دواء الباراسيتامول للأطفا… | 11.4 | 0.01 | · | — | 1 | 5 | 4 | 4 | ✗ |
| 46 | OOC | Who won the 2026 FIFA World Cup? | 15.0 | 0.03 | · | منخفضة | 1 | 2 | 3 | 2 | ✗ |
| 47 | OOC | ما هو سعر البيتكوين اليوم؟ | 2.0 | 0.02 | · | منخفضة | 1 | 2 | 3 | 2 | ✗ |
| 48 | OOC | What are the side effects of ibuprof… | 16.7 | 0.01 | · | منخفضة | 1 | 2 | 2 | 3 | ✗ |
| 49 | OOC | ما هي شروط الحصول على تأشيرة شنغن؟ | 10.3 | 0.00 | · | منخفضة | 1 | 2 | 2 | 3 | ✗ |
| 50 | OOC | How do I file taxes in Egypt? | 15.8 | 0.00 | · | منخفضة | 1 | 2 | 2 | 3 | ✓ |
| 51 | OOC | What is the boiling point of mercury? | 8.8 | 0.01 | · | منخفضة | 1 | 2 | 3 | 3 | ✗ |

## 7) Sample answers

**Q (RAG):** What is Retrieval-Augmented Generation?

```
الإجابة: Retrieval-Augmented Generation (RAG) is a method that enhances the responses generated by large language models (LLMs). It involves several key steps:
1. Retrieval: The query is encoded and compared with precomputed document embeddings to retrieve relevant documents.
2. Ranking: The retrieved documents are ranked based on their relevance to the query.
3. Generation: The top-ranked documents are used as context by the LLM to generate more informed and accurate responses.

The basis for this answer is:
- «Key steps in the Retrieval-Augmented Generation (RAG) pipeline are: 1.Retrieval: The query is encoded and compared with precomputed document embeddings to retrieve relevant documents. 2. Ranking: The retrieved documents are ranked based on their relevance to the query. 3. Generation: The top-ranked documents are used as context by the LLM to generate more informed and accurate re
```

**Q (RAG):** ما هو الـ RAG وكيف يعمل؟

```
الإجابة: راجع Augmented Generation (RAG) هو تقنية تستخدم لتعزيز معرفة النموذج القائم بالتعلم العميق مع بيانات إضافية. يعمل RAG عن طريق استرجاع المعلومات ذات الصلة من مستندات باستخدام التقنيات مثل التشابه السemantic وتقسيم النصوص وإدراجها في عملية إنتاج النص.

الأساس القانوني: لا يوجد سند صريح في المستندات المتاحة.
التسبيب: نظراً لعدم وجود نصوص قانونية محددة تشرح كيفية عمل RAG، فإن الإجابة تعتمد على الوثائق المقدمة التي تصف استخدام التقنية.

درجة الثقة: منخفضة
يحتاج مراجعة بشرية: نعم
```

**Q (OOC):** ما هي عقوبة مخالفة قانون حماية البيانات المصري رقم 151؟

```
الإجابة: لا يوجد سند صريح في المستندات المتاحة يحدد عقوبة مخالفة قانون حماية البيانات المصري رقم 151.
الأساس القانوني: لا يوجد سند صريح في المستندات المتاحة.
التسبيب: نظرًا لعدم وجود نصوص قانونية أو مواد تحدّد العقوبات المترتبة على مخالفات قانون حماية البيانات المصري رقم 151 في المستندات المعطاة، لا يمكن تحديد العقوبة بدقة.
درجة الثقة: منخفضة
يحتاج مراجعة بشرية: نعم
```


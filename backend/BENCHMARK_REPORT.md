# Backend Comparison Report — LanceDB (before) vs Qdrant (after)

- Corpus: **40 questions** over the ingested 9 PDFs (~2,669 chunks)
- Hardware: RTX 5060 Ti 16GB · embed/rerank `cuda` · model `llama3.1:latest`
- Retrieval: hybrid, initial_top_k=40, final_top_k=6
- Generation capped at 300 tokens for stable timing (same for both backends)

## Headline — retrieval latency (what the migration changed)

| Stage | Mean | Median | p95 | Min | Max |
|-------|----:|----:|----:|----:|----:|
| LanceDB — embed (s) | 0.034 | 0.020 | 0.026 | 0.014 | 0.612 |
| LanceDB — DB search (s) | 0.029 | 0.024 | 0.036 | 0.014 | 0.210 |
| LanceDB — rerank (s) | 0.483 | 0.461 | 0.600 | 0.324 | 1.540 |
| **LanceDB — retrieval total (s)** | 0.546 | 0.502 | 0.647 | 0.365 | 2.362 |
| Qdrant — embed (s) | 0.020 | 0.020 | 0.025 | 0.016 | 0.025 |
| Qdrant — DB search (s) | 0.043 | 0.042 | 0.049 | 0.036 | 0.050 |
| Qdrant — rerank (s) | 0.478 | 0.467 | 0.736 | 0.320 | 0.740 |
| **Qdrant — retrieval total (s)** | 0.541 | 0.529 | 0.796 | 0.390 | 0.799 |

## Pure DB-search latency (the apples-to-apples bit)

- LanceDB mean DB search: **29.0 ms**
- Qdrant  mean DB search: **42.7 ms**
- Difference: **+13.7 ms** per query (LanceDB faster)

## End-to-end (retrieval + generation)

| Backend | Mean total (s) | Median | p95 | Retrieval share |
|---------|----:|----:|----:|----:|
| LanceDB | 5.22 | 5.30 | 5.79 | 10.5% |
| Qdrant  | 5.22 | 5.32 | 5.85 | 10.4% |

- Shared generation: mean **4.67s** (40 questions)
- **Generation is 90% of total time** — the vector DB is a small fraction.

## Retrieval agreement (quality sanity check)

- Mean Jaccard overlap of the final top-6 chunks (LanceDB vs Qdrant): **87%** — both hybrid retrievers surface largely the same evidence.

## Per-question (retrieval total, seconds)

| # | Question | LanceDB | Qdrant | DB search L→Q (ms) | overlap |
|---|----------|----:|----:|----:|----:|
| 1 | What is Retrieval-Augmented Generat... | 2.362 | 0.460 | 210→48 | 100% |
| 2 | How does the retrieval step in RAG ... | 0.400 | 0.440 | 24→45 | 100% |
| 3 | What is reranking in a RAG pipeline? | 0.465 | 0.480 | 25→49 | 71% |
| 4 | ما هو الـ RAG وكيف يعمل؟ | 0.468 | 0.517 | 30→42 | 100% |
| 5 | What is chunking and why does it ma... | 0.557 | 0.562 | 34→37 | 100% |
| 6 | What is a transformer architecture? | 0.647 | 0.665 | 27→46 | 100% |
| 7 | What is the attention mechanism? | 0.433 | 0.435 | 25→39 | 100% |
| 8 | What is a large language model? | 0.546 | 0.554 | 29→38 | 71% |
| 9 | ما هو نموذج اللغة الكبير LLM؟ | 0.460 | 0.456 | 26→41 | 71% |
| 10 | What is the context window in an LLM? | 0.474 | 0.543 | 25→42 | 71% |
| 11 | What is tokenization in NLP? | 0.522 | 0.539 | 24→36 | 100% |
| 12 | What is a word embedding? | 0.556 | 0.450 | 26→40 | 100% |
| 13 | What is the difference between fine... | 0.470 | 0.474 | 30→44 | 71% |
| 14 | What are the differences between LL... | 0.436 | 0.467 | 27→40 | 71% |
| 15 | ما هي آلية الانتباه attention؟ | 0.454 | 0.494 | 24→43 | 71% |
| 16 | How do you build a reliable AI agent? | 0.365 | 0.425 | 21→45 | 100% |
| 17 | How do you evaluate an AI agent? | 0.365 | 0.390 | 27→50 | 100% |
| 18 | What are common ways to build AI ag... | 0.405 | 0.424 | 24→41 | 100% |
| 19 | ما هي طرق بناء وكلاء الذكاء الاصطناعي؟ | 0.500 | 0.796 | 14→40 | 100% |
| 20 | What is a Docker container? | 0.493 | 0.538 | 36→46 | 100% |
| 21 | What is the difference between a Do... | 0.504 | 0.524 | 25→42 | 100% |
| 22 | What is a Dockerfile? | 0.505 | 0.530 | 29→41 | 100% |
| 23 | ما هو الـ Docker container؟ | 0.490 | 0.500 | 16→43 | 100% |
| 24 | What is a primary key in SQL? | 0.610 | 0.630 | 26→46 | 71% |
| 25 | What is a foreign key? | 0.605 | 0.615 | 24→41 | 71% |
| 26 | What is the difference between DELE... | 0.512 | 0.623 | 22→47 | 100% |
| 27 | What does the JOIN keyword do? | 0.561 | 0.765 | 32→45 | 100% |
| 28 | How do you create a table with a pr... | 0.611 | 0.638 | 20→47 | 71% |
| 29 | ما هو المفتاح الأساسي في SQL؟ | 0.521 | 0.537 | 21→45 | 71% |
| 30 | What is an SQL index? | 0.494 | 0.528 | 24→38 | 100% |
| 31 | What is named entity recognition? | 0.510 | 0.540 | 21→45 | 100% |
| 32 | What is transfer learning in NLP? | 0.460 | 0.472 | 22→40 | 100% |
| 33 | What is a vector database used for? | 0.488 | 0.505 | 25→41 | 71% |
| 34 | What is semantic search? | 0.530 | 0.438 | 23→42 | 71% |
| 35 | ما هو البحث الدلالي؟ | 0.508 | 0.799 | 18→40 | 71% |
| 36 | What is multimodality in language m... | 0.550 | 0.566 | 23→40 | 100% |
| 37 | What is the Querying Transformer (Q... | 0.473 | 0.488 | 28→45 | 71% |
| 38 | What is the difference between an e... | 0.527 | 0.536 | 16→46 | 100% |
| 39 | What is quantization of a model? | 0.490 | 0.747 | 23→38 | 20% |
| 40 | ما الفرق بين الـ encoder و الـ deco... | 0.529 | 0.535 | 16→46 | 71% |

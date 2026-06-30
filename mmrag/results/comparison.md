## Pipeline Comparison — Retrieval + Quality (RAGAS) (85 questions)

| Metric | Baseline | Optimized | Δ |
|--------|:--------:|:---------:|:---:|
| Hit@1 | 0.953 | 0.965 | +0.012 |
| Hit@3 | 1.000 | 1.000 | +0.000 |
| Hit@5 | 1.000 | 1.000 | +0.000 |
| Recall@5 | 1.000 | 1.000 | +0.000 |
| Recall@10 | 1.000 | 1.000 | +0.000 |
| nDCG@10 | 0.983 | 0.987 | +0.004 |
| MRR | 0.976 | 0.982 | +0.006 |
| Latency p50 (ms) | 2726 | 2994 | +268 |
| Latency p95 (ms) | 3050 | 3442 | +391 |
| **— RAGAS quality —** | | | |
| Faithfulness | 0.628 | 0.684 | +0.056 |
| Answer Relevancy | 0.576 | 0.659 | +0.083 |
| Context Recall | 0.569 | 0.563 | -0.006 |
| Answer Correctness | 0.638 | 0.623 | -0.014 |

**Baseline** : recursive chunking · dense-only retrieval · no reranking  
**Optimized**: semantic chunking · BM25 + dense hybrid (RRF) · CrossEncoder reranking  
*Embedding models — baseline: `all-MiniLM-L6-v2` · optimized: `BAAI/bge-small-en-v1.5`*
*RAGAS judge: local Ollama · embeddings: `all-MiniLM-L6-v2`*

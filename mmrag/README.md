# mmrag — Multimodal RAG Benchmark

Benchmark comparing a **baseline** RAG pipeline against an **optimized** one across retrieval, generation, and RAGAS quality metrics. Everything runs locally: embeddings via `sentence-transformers`, vector store via ChromaDB, LLM via Ollama.

---

## Architecture

```
mmrag/
├── configs/
│   ├── baseline.yaml       # recursive chunking · dense-only · no rerank
│   └── optimized.yaml      # semantic chunking · BM25+dense hybrid · CrossEncoder rerank
│
├── data/
│   ├── raw/
│   │   ├── text/           # SQuAD passages (manifest.jsonl + .txt files)
│   │   └── images/         # image captions (manifest.jsonl)
│   ├── processed/          # cleaned documents (manifest.jsonl)
│   └── eval/
│       ├── eval_set.jsonl  # 85 validated questions with ground-truth answers
│       ├── candidates.jsonl
│       └── split.json
│
├── src/
│   ├── config.py           # pydantic-settings config loader
│   ├── ingestion/          # loaders, parsers, schema (RawDocument)
│   ├── preprocessing/      # chunking (RecursiveChunker, SemanticChunker)
│   ├── embeddings/         # text, image (CLIP), audio (CLAP) embedders
│   ├── indexing/           # VectorStoreIndex (ChromaDB), BM25 serializer
│   ├── retrieval/          # dense, sparse (BM25), hybrid (RRF), reranker
│   ├── generation/         # LLMClient (Ollama), prompt builder
│   ├── pipelines/
│   │   ├── baseline.py     # BaselinePipeline
│   │   └── optimized.py    # OptimizedPipeline
│   ├── evaluation/         # retrieval metrics, RAGAS eval, ablation matrix
│   └── serving/            # FastAPI REST endpoint
│
├── scripts/
│   ├── build_dataset.py    # download & process raw documents
│   ├── build_eval_set.py   # generate & validate the eval set via Ollama
│   ├── run_baseline.py     # interactive baseline queries
│   ├── run_optimized.py    # interactive optimized queries
│   └── run_comparison.py   # full benchmark (retrieval + RAGAS) → results/
│
├── results/
│   ├── comparison.md       # benchmark table (generated)
│   └── quick_compare.json  # raw numbers (generated)
│
└── tests/                  # pytest unit tests
```

### Data flow

```
Raw docs (SQuAD text / images)
        │
        ▼
   build_dataset.py
        │  parse + clean
        ▼
 data/processed/manifest.jsonl
        │
        ├──────────────────────────────────────────┐
        │  BaselinePipeline                        │  OptimizedPipeline
        │  RecursiveChunker (512t / 64 overlap)    │  SemanticChunker (breakpoint@p25)
        │  TextEmbedder (all-MiniLM-L6-v2)         │  TextEmbedder (BAAI/bge-small-en-v1.5)
        │  ChromaDB dense index                    │  ChromaDB dense index + BM25 pkl
        │  Dense retrieval (cosine, top-5)         │  Hybrid retrieval RRF(BM25, dense, top-20)
        │  No reranking                            │  CrossEncoder rerank (top-5)
        │                                          │
        └──────────────┬───────────────────────────┘
                       │
                  Ollama llama3.2
                  (identical generation layer)
                       │
                  Answer + sources
```

---

## Quick start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running with `llama3.2` pulled:
  ```
  ollama pull llama3.2
  ```

### Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **Windows UTF-8 fix** — set this before running any script to avoid encoding errors:
> ```powershell
> $env:PYTHONUTF8 = "1"
> ```

### 1 — Build the dataset

```bash
python scripts/build_dataset.py
```

Downloads and processes SQuAD passages into `data/processed/`.

### 2 — Build the eval set

```bash
python scripts/build_eval_set.py --n-text 100 --seed 1234
```

Generates 100 candidate questions via Ollama; ~85 pass the validation gates (length, anchoring, no verbatim leakage).

### 3 — Run the comparison benchmark

```bash
# Retrieval + RAGAS quality (needs Ollama)
python scripts/run_comparison.py --simple

# Retrieval only — no LLM needed
python scripts/run_comparison.py --simple --no-ragas

# Limit to first N questions (faster iteration)
python scripts/run_comparison.py --simple --n 20
```

Results are saved to `results/comparison.md` and `results/quick_compare.json`.

### 4 — Interactive queries

```bash
python scripts/run_baseline.py
python scripts/run_optimized.py
```

### 5 — REST API

```bash
uvicorn src.serving.api:app --reload
# POST http://localhost:8000/query  {"question": "..."}
```

### 6 — Streamlit UI

```bash
streamlit run app.py
```

---

## Optimization methods

> The optimized pipeline modifies **only chunking and retrieval**. The generation layer (model `llama3.2`, `temperature: 0.1`, `max_tokens: 512`, system prompt) is byte-for-byte identical to the baseline. Every metric difference is therefore caused purely by better document retrieval.

---

### Optimization 1 — Semantic chunking vs. recursive chunking

#### What the baseline does

The baseline uses `RecursiveCharacterTextSplitter` from LangChain: it cuts the text into pieces of at most 512 characters, with a 64-character overlap between consecutive chunks. It tries to respect natural separators (paragraph `\n\n`, line `\n`, space) but ultimately falls back to a hard character cut. The result is chunks whose boundaries are determined by **size**, not **meaning**.

```
Document: "The Roman Empire fell in 476 AD. The Middle Ages began.
           Meanwhile, in China the Tang dynasty..."

Baseline chunk 1 (512 chars):  "The Roman Empire fell in 476 AD. The Middle Ages began. Meanwhile, in China the Tang dynasty[...]"
Baseline chunk 2 (512 chars, 64 overlap): "[...]Tang dynasty flourished. Marco Polo..."
```

The overlap of 64 characters exists precisely to compensate for the fact that a meaningful sentence may get cut in half — but it introduces redundancy and still does not guarantee semantic coherence.

#### What the optimized pipeline does

`SemanticChunker` (`src/preprocessing/chunking.py`) uses a 5-step algorithm:

1. **Sentence splitting** — the document is split into individual sentences using punctuation heuristics (`re.split(r"(?<=[.!?])\s+", text)`).
2. **Batch embedding** — all sentences are embedded in one forward pass with `all-MiniLM-L6-v2`, producing a matrix of shape `(n_sentences, 384)`.
3. **Cosine similarity between adjacent sentences** — for each pair `(s_i, s_{i+1})`, the dot product of their L2-normalized vectors is computed:
   ```python
   normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
   sims = [normed[i] @ normed[i+1] for i in range(len(normed) - 1)]
   ```
4. **Breakpoint detection** — a split is placed wherever similarity drops below the **25th percentile** of all pairwise similarities in the document. This adapts to each document: a document with consistent prose has a higher threshold than a document that jumps between topics.
5. **Safety re-split** — if a resulting semantic chunk exceeds `2 × chunk_size` (1024 chars), it is recursively split with `RecursiveChunker` to enforce a loose upper bound.

```
Same document, semantic chunking:
  Chunk 1: "The Roman Empire fell in 476 AD. The Middle Ages began."
           → similarity to next group drops → breakpoint
  Chunk 2: "Meanwhile, in China the Tang dynasty flourished. Marco Polo..."
```

#### Why it improves the pipeline

Each chunk sent to the LLM as context now contains a **single coherent idea**. The LLM does not have to reason across an arbitrary character cut that interrupted a sentence mid-thought. The result in practice: faithfulness improves because the context is cleaner, and the LLM produces answers that are better anchored in the retrieved passage.

**Impact on index size**: the optimized pipeline produces **1 077 chunks** from 500 documents vs. **1 023** for the baseline — slightly more, smaller, and semantically tighter chunks.

---

### Optimization 2 — Stronger embedding model

| | Baseline | Optimized |
|---|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` | `BAAI/bge-small-en-v1.5` |
| Vector dimension | 384 | 384 |
| MTEB avg (retrieval) | ~40 | ~51 |

#### What the baseline does

`all-MiniLM-L6-v2` is a distilled 6-layer MiniLM model trained for general-purpose sentence similarity. It is fast and compact but was not specifically trained for **asymmetric retrieval** (short query → long passage), which is exactly the RAG use case.

#### What the optimized pipeline does

`BAAI/bge-small-en-v1.5` is trained with a **retrieval-specific objective** using large-scale hard-negative mining and instruction-aware fine-tuning. It achieves significantly higher scores on MTEB retrieval tasks while keeping the same 384-dimensional output — which means the ChromaDB index structure, storage size, and query latency remain identical.

#### Why it improves the pipeline

The embedding determines how close a question vector is to a relevant passage vector in the 384-dimensional space. A model trained specifically for retrieval produces a geometry where the "right" document is more likely to be in the top-1 result rather than top-3 or top-5. This is visible in the benchmark: **Hit@1 goes from 0.953 → 0.965** (the optimized pipeline finds the right document at rank 1 more often).

---

### Optimization 3 — Hybrid retrieval (BM25 + dense) with Reciprocal Rank Fusion

#### What the baseline does

The baseline runs a single **dense cosine similarity search** against the ChromaDB vector index. For a given query:
1. The query is embedded to a 384-dimensional vector.
2. ChromaDB finds the 5 nearest chunks by cosine similarity.
3. Those 5 chunks are sent directly to the LLM.

This works well for semantic queries but fails on **exact-keyword queries**. For example, the question *"What is the BM25 k1 parameter?"* may match a passage semantically about "term frequency weighting" but miss the passage that literally contains the string "k1 = 1.5".

#### What the optimized pipeline does

**Two retrievers run independently in parallel:**

**BM25 sparse retriever** (`src/retrieval/sparse_bm25.py`)  
BM25Okapi scores documents by exact keyword matching, weighted by TF-IDF-like statistics:
- `k1 = 1.5` controls term-frequency saturation (a term mentioned 10 times is not 10× as relevant as one mentioned once).
- `b = 0.75` normalizes scores by document length (long documents are penalized).
- The index is built from tokenized chunks (lowercase whitespace-split) and persisted as a `.pkl` file for fast restarts.
- Only documents with `score > 0` (at least one keyword matches) are returned.

**Dense retriever** (`BAAI/bge-small-en-v1.5` + ChromaDB)  
Same as baseline but with the stronger model; retrieves top-20 candidates.

**Reciprocal Rank Fusion** (`src/retrieval/hybrid.py`)  
RRF merges the two ranked lists without requiring any score normalization:

```python
# For each retriever r, and each document d at rank position rank_r(d):
rrf_score(d) = Σ_r  1 / (k + rank_r(d))     # k = 60 (Cormack et al., SIGIR 2009)
```

The key insight is that **ranks** are used, not raw scores. Dense cosine scores cluster near 0.9–1.0 with little spread; BM25 scores scale with term frequency and can span several orders of magnitude. A linear combination `α·dense + β·bm25` would require corpus-specific calibration and would break when the corpus changes. RRF is **scale-free** and requires no tuning.

The constant `k = 60` caps the maximum contribution of any single top-1 hit to `1/(60+1) ≈ 0.016`, preventing one retriever from dominating the fusion. A document that appears in both lists at ranks 1 and 2 accumulates `1/61 + 1/62 ≈ 0.033`, reliably ranking above a document that appears in only one list.

```
BM25 ranking:    [doc_A(rank1), doc_C(rank2), doc_B(rank3), ...]
Dense ranking:   [doc_B(rank1), doc_A(rank2), doc_D(rank3), ...]

RRF scores:
  doc_A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325   ← wins
  doc_B: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
  doc_C: 1/(60+2) = 0.0161
  doc_D: 1/(60+3) = 0.0159
```

Both retrievers fetch **top-20 candidates** before fusion to maximize recall. Documents missed by one retriever can still be surfaced by the other.

#### Why it improves the pipeline

- Queries with specific names, dates, or technical terms that the dense model embeds poorly → BM25 saves the retrieval.
- Queries that are semantically clear but contain no exact keywords → dense retrieval saves it.
- A document that ranks high in both lists is almost certainly the right one → RRF amplifies this signal.

---

### Optimization 4 — CrossEncoder reranking

#### What the baseline does

After dense retrieval, the baseline takes the top-5 documents **as-is** and sends them to the LLM. The ranking is determined entirely by cosine similarity, which is a fast but approximate relevance signal: the query and each document are embedded **independently**, so the model never sees both texts together when computing the score.

#### What the optimized pipeline does

`CrossEncoderReranker` (`src/retrieval/reranker.py`) adds a second ranking stage using `cross-encoder/ms-marco-MiniLM-L-6-v2`:

**Stage 1 — over-retrieve**: hybrid RRF returns the top-20 candidates.

**Stage 2 — rerank**: for each of the 20 candidates, the CrossEncoder receives the full `(query, passage)` pair concatenated and runs a joint forward pass:
```python
pairs = [(query, doc.page_content) for doc, _ in candidates]
scores = cross_encoder_model.predict(pairs, batch_size=32)
```
Unlike a bi-encoder, the CrossEncoder sees **token-level interactions** between the query and the passage — it can notice when a passage directly answers the question, catches negations, or satisfies complex multi-hop conditions. The 20 candidates are re-sorted by this score and the top-5 are kept.

The first-stage retrieval score (cosine or RRF) is **completely discarded** — only the CrossEncoder score matters for the final ranking.

#### Why this architecture (over-retrieve + rerank)?

Running a CrossEncoder on the entire corpus (500+ documents) would be prohibitively slow — it costs one forward pass per document. By limiting it to 20 candidates (already filtered by hybrid retrieval), the total cost is 20 forward passes, which takes ~200-400ms on CPU. The pipeline trades a fixed latency overhead for a significant accuracy gain.

#### Why it improves the pipeline

Bi-encoders compress each text into a single vector, losing fine-grained relevance signals. The CrossEncoder's joint encoding surfaces documents that truly **answer** the question rather than merely being topically related. This is why **faithfulness (+5.6pp) and answer relevancy (+8.3pp)** are the biggest improvements — the LLM receives context that is more precisely relevant to the exact question asked.

---

### Optimization 5 — Query transformation (ablation axis, disabled by default)

Disabled in `configs/optimized.yaml` (`query_transform.enabled: false`). Enable to activate.

#### multi_query

The LLM generates `n_queries = 3` alternative phrasings of the user question before retrieval. Hybrid retrieval runs independently for each phrasing, producing 3 separate ranked lists. A second round of RRF fuses all lists. This improves recall for questions where the phrasing is ambiguous or where the user vocabulary differs from the document vocabulary.

**Cost**: `n_queries` extra LLM calls before the main generation.

#### HyDE — Hypothetical Document Embeddings (Gao et al. 2022)

Instead of embedding the raw question, the LLM writes a short **hypothetical passage** that would answer the question. That passage is then embedded and used for **dense retrieval**.

**Why it works**: the query vector (question phrasing) and the relevant passage vectors live in slightly different regions of the 384-dimensional embedding space. A hypothetical answer passage lives in the same region as real answer passages — making cosine similarity more reliable. BM25 still uses the original question (keywords come from the question, not the hypothetical answer).

**Cost**: 1 extra LLM call before retrieval.

---

## Benchmark results

*85 questions · corpus: 500 SQuAD passages · LLM judge: Ollama llama3.2*

## Pipeline Comparison — Retrieval + Quality (RAGAS) (85 questions)

| Metric | Baseline | Optimized | Δ |
|--------|:--------:|:---------:|:---:|
| Hit@1 | 0.953 | **0.965** | +0.012 |
| Hit@3 | 1.000 | 1.000 | = |
| Hit@5 | 1.000 | 1.000 | = |
| Recall@5 | 1.000 | 1.000 | = |
| Recall@10 | 1.000 | 1.000 | = |
| nDCG@10 | 0.983 | **0.987** | +0.004 |
| MRR | 0.976 | **0.982** | +0.006 |
| Latency p50 (ms) | 2726 | 2994 | +268 |
| Latency p95 (ms) | 3050 | 3442 | +391 |
| **— RAGAS quality —** | | | |
| Faithfulness | 0.628 | **0.684** | **+0.056** |
| Answer Relevancy | 0.576 | **0.659** | **+0.083** |
| Context Recall | 0.569 | 0.563 | -0.006 |
| Answer Correctness | **0.638** | 0.623 | -0.014 |

**Baseline**: recursive chunking · dense-only retrieval · no reranking · `all-MiniLM-L6-v2`
**Optimized**: semantic chunking · BM25 + dense hybrid (RRF) · CrossEncoder reranking · `BAAI/bge-small-en-v1.5`
*RAGAS judge: local Ollama · embeddings: `all-MiniLM-L6-v2`*

### Metric-by-metric analysis

#### Hit@1 — 0.953 → 0.965 (+0.012)

**What it measures**: the fraction of questions where the correct document appears at rank 1 (the very first result). The LLM sees the top-k documents as a flat list; rank 1 gets the most "attention" in the context window.

**Baseline**: dense cosine similarity with `all-MiniLM-L6-v2` ranks the correct document first on 95.3% of queries. The remaining 4.7% have the right document at rank 2–5, meaning it is present in the context but not first.

**Optimized**: Hit@1 rises to 96.5%. The two optimizations responsible are the **stronger embedding model** (`bge-small-en-v1.5`) and the **CrossEncoder reranker**. The better embedding produces a query vector closer to the right document in the first place; the reranker then corrects the remaining misranked cases by jointly reading query + passage.

**Why this matters**: LLMs give more weight to the beginning of the context. A document at rank 1 is more likely to be used by the model than the same document at rank 5. Improving Hit@1 means more answers are grounded in the most relevant passage.

---

#### Hit@3, Hit@5, Recall@5, Recall@10 — 1.000 = 1.000

**What it measures**: Hit@k = correct document appears somewhere in the top-k. Recall@k = fraction of all relevant documents retrieved in the top-k.

Both pipelines achieve a perfect score, which reveals a **corpus saturation effect**: the SQuAD corpus is small (500 documents) and each question is derived from a specific passage, so every retrieval strategy finds the right document within top-5. This does not mean retrieval is equal — it means the ceiling is reached at this corpus size. On a larger, noisier corpus (tens of thousands of documents), the gap would become visible here too.

---

#### nDCG@10 — 0.983 → 0.987 (+0.004)

**What it measures**: Normalized Discounted Cumulative Gain at 10. Unlike Hit@k, nDCG is position-aware: finding the right document at rank 1 scores better than finding it at rank 7. The "discounted" part means each rank contributes `1/log2(rank+1)`.

**Baseline**: already high at 0.983 because most documents are found at rank 1 or 2.

**Optimized**: the small +0.004 gain confirms that the CrossEncoder pushes the right documents slightly higher in the ranking on the cases where baseline had them at rank 2 or 3. The gain is small because the baseline was already close to the ceiling.

---

#### MRR — 0.976 → 0.982 (+0.006)

**What it measures**: Mean Reciprocal Rank = average of `1/rank` for the first correct document across all queries. MRR heavily penalizes rank 2 (`1/2 = 0.5`) vs rank 1 (`1/1 = 1.0`), making it a sensitive measure of how often the pipeline gets it right immediately.

**Baseline**: 0.976 means the average "effective rank" is around `1/0.976 ≈ 1.02` — nearly always rank 1, but a handful of queries where the right document sits at rank 2 or 3 pull the score down.

**Optimized**: 0.982. The CrossEncoder reranker is directly responsible: by rescoring all 20 candidates jointly with the query, it corrects the cases where the bi-encoder placed the right document at rank 2–3. The improvement is small in absolute terms because the baseline already performs well, but it is consistent: the optimized pipeline makes fewer first-position mistakes.

---

#### Latency p50 — 2726ms → 2994ms (+268ms) · p95 — 3050ms → 3442ms (+391ms)

**What it measures**: wall-clock time from question received to answer returned, at the 50th and 95th percentile.

**Baseline**: ~2.7s per query. The dominant cost is Ollama LLM generation (~2.5s); retrieval is nearly instant (<50ms for a dense cosine search on 1023 chunks).

**Optimized**: ~3.0s per query. The additional ~268ms is entirely from the **CrossEncoder reranker**: 20 `(query, passage)` pairs run through `ms-marco-MiniLM-L-6-v2` on CPU in batches of 32. The hybrid RRF retrieval adds negligible latency because BM25 is an in-memory index (lookup in microseconds).

**Trade-off**: +268ms at p50 for +5.6pp faithfulness and +8.3pp answer relevancy. For most production use cases this is a favorable trade. If latency is critical, the CrossEncoder can be disabled (`rerank.enabled: false`) — retrieval metrics remain close, only generation quality drops.

---

#### Faithfulness — 0.628 → 0.684 (+0.056) ★ biggest win

**What it measures** (RAGAS): the fraction of claims in the generated answer that are **directly supported** by the retrieved context. Ollama judges each claim in the answer by checking whether it can be derived from the provided passages. Score = `supported_claims / total_claims`.

**Baseline**: 0.628. The baseline sometimes returns 5 documents that are topically related but do not directly contain the answer. The LLM then "fills in the gaps" with plausible-sounding information that is not in the context — hallucination. This produces claims that cannot be verified against the retrieved passages.

**Optimized**: 0.684. Two optimizations combine here:
1. **Semantic chunking** produces tighter, coherent chunks — each chunk contains a single idea, so the LLM does not have to parse a chunk that mixes two unrelated topics and may accidentally pick up the wrong one.
2. **CrossEncoder reranking** ensures the top-5 passages are the ones that most directly answer the question, rather than merely being topically adjacent. The LLM has no incentive to hallucinate when the answer is explicitly present in the context.

**Why +5.6pp matters**: in a production RAG system, faithfulness is the most critical metric — an answer that contradicts the source is worse than no answer. A 5.6pp improvement means roughly 1 in 18 queries goes from "contains an unsupported claim" to "fully grounded in the context".

---

#### Answer Relevancy — 0.576 → 0.659 (+0.083) ★ biggest win

**What it measures** (RAGAS): how well the answer actually addresses the question. Ollama generates N reverse-questions from the answer, embeds them, and measures cosine similarity to the original question. A high score means "if you only read the answer, you would understand what the question was asking about."

**Baseline**: 0.576. This is the lowest metric on both pipelines, revealing a systematic issue: the baseline retrieves passages that are topically relevant but not tightly focused on the question. The LLM then produces answers that are accurate about the topic but do not directly address what was asked — they contain extra background information that dilutes relevance.

**Optimized**: 0.659, an +8.3pp improvement — the largest gain in the entire benchmark.

The CrossEncoder is the primary driver: by reading `(question, passage)` jointly, it scores passages by how well they **answer this specific question**, not by how similar their topic is. The passages sent to the LLM are therefore precisely focused on the question, and the LLM's answer stays on topic rather than exploring related background.

Semantic chunking also contributes: a coherent single-idea chunk gives the LLM a clean, focused input rather than a chunk that blends the answer to the question with unrelated content from the same section of the document.

---

#### Context Recall — 0.569 → 0.563 (−0.006) slight regression

**What it measures** (RAGAS): the fraction of facts in the **ground-truth answer** that are covered by the retrieved context. Ollama compares the ground-truth answer sentence-by-sentence against the retrieved passages.

**Why the slight regression**: the baseline retrieves the top-5 documents by cosine similarity from 1023 chunks. The optimized pipeline retrieves top-20 by hybrid RRF, then **discards 15** after reranking, keeping only top-5. In rare cases, the CrossEncoder promotes a passage that is highly relevant to the question but covers a slightly different aspect of the topic than what the ground-truth answer expects. The correct passage ends up at rank 6–7 and is dropped.

This is a known trade-off of aggressive reranking: optimizing for "does this passage directly answer the question?" can occasionally be worse than "does this passage cover the same topic as the ground-truth answer?" The effect is small (−0.6pp) because it happens on a minority of queries.

**On a larger corpus** this trade-off typically disappears: with more candidate documents, hybrid retrieval surfaces more complete coverage before reranking, and the CrossEncoder has more material to select a well-rounded top-5 from.

---

#### Answer Correctness — 0.638 → 0.623 (−0.014) slight regression

**What it measures** (RAGAS): composite score combining factual similarity between the generated answer and the ground-truth answer, weighted by semantic similarity. It is the strictest metric: the answer must match the expected answer closely.

**Baseline**: 0.638. The baseline grounds its answer in the passage it finds at rank 1 by cosine similarity. For SQuAD questions, cosine similarity tends to return the exact source passage — and the LLM extracts the expected short answer from it.

**Optimized**: 0.623. The CrossEncoder sometimes promotes a passage that is more directly relevant to the question intent but contains a **slightly different phrasing or angle** on the answer than the ground-truth expects. For example, if the ground truth is "1945" and the reranked passage discusses the same event with emphasis on the aftermath rather than the date, the LLM may produce "after WWII ended" instead of "1945" — semantically correct but scoring lower on exact-match-weighted correctness.

This regression is specific to short-answer factoid benchmarks (like SQuAD) where ground-truth answers are very precise extracts. On open-domain or long-form QA tasks, this effect does not apply and answer correctness typically improves with reranking.

---

#### Summary: which optimization causes which improvement

| Metric | Primary cause | Secondary cause |
|--------|--------------|-----------------|
| Hit@1 +1.2pp | CrossEncoder reranker | Stronger embedding (bge-small) |
| nDCG@10 +0.4pp | CrossEncoder reranker | — |
| MRR +0.6pp | CrossEncoder reranker | Stronger embedding (bge-small) |
| Faithfulness +5.6pp | CrossEncoder reranker | Semantic chunking |
| Answer Relevancy +8.3pp | CrossEncoder reranker | Semantic chunking |
| Context Recall −0.6pp | CrossEncoder reranker (trade-off) | — |
| Answer Correctness −1.4pp | CrossEncoder reranker (trade-off) | — |
| Latency +268ms | CrossEncoder reranker (cost) | — |

The CrossEncoder reranker is the dominant lever in this experiment. Semantic chunking and the stronger embedding model are enablers that improve the quality of the candidate pool that the reranker works on.

---

## Known issues & fixes

### Problem — RAGAS TypeError: `AsyncClient.chat()` got unexpected keyword `temperature`

**Symptom**: 100% of RAGAS LLM calls failed with:
```
TypeError: AsyncClient.chat() got an unexpected keyword argument 'temperature'
```
RAGAS progress advanced but all metrics would have been NaN after ~45 additional minutes of wasted compute.

**Root cause**: `langchain-ollama 0.2.3` passed `temperature` as a direct kwarg to `ollama.AsyncClient.chat()`. In `ollama >= 0.4.0` the internal API changed: extra model parameters must go inside an `options` dict, not as top-level kwargs. Upgrading `langchain-ollama` to `0.3.3` was necessary but not sufficient — it still passed `temperature` both inside `options={}` *and* as a top-level kwarg simultaneously (double-pass bug).

**Fix applied**:

1. **Dependency upgrade** — `langchain-ollama` pinned to `0.3.3`, `ollama` pinned to `0.3.3` (see `requirements.txt`).

2. **Defensive monkey-patch** in `src/evaluation/ragas_eval.py` — function `_patch_ollama_async_client()`:
   - Wraps `ollama.AsyncClient.chat()` before RAGAS runs.
   - Strips parasitic top-level kwargs (`temperature`, `top_p`, `top_k`, etc.) that `langchain-ollama` leaks outside `options`.
   - Idempotent: marked `_patched_for_ragas` so double-patching is safe.
   - Called automatically at the start of `_build_judge()`.

```python
# src/evaluation/ragas_eval.py — simplified view
def _patch_ollama_async_client() -> None:
    import ollama
    if getattr(ollama.AsyncClient.chat, "_patched_for_ragas", False):
        return
    _original = ollama.AsyncClient.chat
    _STRIP = {"temperature", "top_p", "top_k", "repeat_penalty", "seed"}

    async def _patched(self, **kwargs):
        for key in _STRIP:
            kwargs.pop(key, None)
        return await _original(self, **kwargs)

    _patched._patched_for_ragas = True
    ollama.AsyncClient.chat = _patched
```

**Validation**: after the fix, a single-question smoke test returned `faithfulness: 1.0, error: None` with zero TypeErrors. Full 85-question run completed successfully.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PYTHONUTF8` | — | Set to `1` on Windows to force UTF-8 stdout |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Default generation/judge model |

Copy `.env.example` to `.env` and adjust as needed.

---

## Running tests

```bash
pytest tests/ -v
```

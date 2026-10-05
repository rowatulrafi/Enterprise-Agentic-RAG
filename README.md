# Enterprise Agentic RAG Pipeline

A fully local Retrieval-Augmented Generation system for document-heavy enterprise knowledge workflows.

The pipeline combines native/OCR document ingestion, hybrid dense + keyword retrieval, cross-encoder reranking, agentic query rewriting, parent-child context retrieval, and citation-grounded answer verification.

**Stack:** Python · LangGraph · LangChain · ChromaDB · BM25 · Sentence Transformers · Qwen3-8B · LM Studio · Tesseract OCR

---

## Evaluation Highlights

| Evaluation | Result | Purpose |
|---|---:|---|
| Retrieval development benchmark | **94% Hit@3** | Retrieval tuning |
| Locked unseen end-to-end benchmark | **80% strict pass rate** | Generalization |
| Substantive incorrect answers exposed on unseen benchmark | **0 / 20** | Safety behavior |

The final 20-query end-to-end benchmark was locked before execution and preserved with a SHA-256 checksum.

Its untouched first run produced:

- **16 / 20 full reference passes**
- **2 / 20 partially correct but incomplete answers**
- **1 / 20 safe abstention**
- **1 / 20 context-window runtime failure**
- **0 / 20 substantive incorrect answers returned to the user**

> The 94% Hit@3 result comes from a development retrieval benchmark and should not be interpreted as end-to-end answer accuracy.

---

## What the System Does

The project is designed around four goals:

- retrieve both semantic and exact-match evidence;
- preserve useful document context without sacrificing retrieval precision;
- recover from weak retrieval through controlled query rewriting;
- prevent unsupported generated claims from reaching the user.

### Core capabilities

- Native PDF text extraction with OCR fallback
- LLM-assisted OCR quality validation
- Human-review queue for low-confidence extraction
- Dense vector retrieval with ChromaDB
- Sparse BM25 keyword retrieval
- Reciprocal candidate merging and cross-encoder reranking
- Parent-child retrieval with layout-preserved page hydration
- Retrieval sufficiency grading
- One-step agentic query rewriting
- Local Qwen3 inference through LM Studio
- Citation-local grounding verification
- Safe abstention for unsupported core answers
- Deterministic exact-substring removal of unsupported optional claims
- Per-stage latency telemetry
- Locked evaluation datasets and preserved first-run results

---

## Architecture

```text
                         PDF Documents
                              |
                              v
                   Native Text Extraction
                              |
                 +------------+------------+
                 |                         |
          usable native text         weak / scanned page
                 |                         |
                 |                         v
                 |                     OCR Extraction
                 |                         |
                 |                  OCR Quality Check
                 |                         |
                 |              +----------+----------+
                 |              |                     |
                 |            pass                 low confidence
                 |              |                     |
                 |              |                     v
                 |              |              Human Review Queue
                 |              |
                 +--------------+
                        |
                        v
                Structured Page Corpus
                        |
              +---------+---------+
              |                   |
              v                   v
        Dense Retrieval       BM25 Retrieval
              |                   |
              +---------+---------+
                        |
                        v
               Hybrid Candidate Set
                        |
                        v
              Cross-Encoder Reranking
                        |
                        v
             Parent-Page Context Hydration
                        |
                        v
               Sufficiency Grading
                  |             |
             sufficient     insufficient
                  |             |
                  |             v
                  |        Query Rewrite
                  |             |
                  +------> Retrieval
                        |
                        v
                Grounded Generation
                        |
                        v
              Citation-Local Verification
                   |                |
                 pass             fail
                   |                |
                   v                v
                 Answer       Safe Fallback
                              or Exact Redaction
```

---

## Key Design Decisions

### 1. Hybrid retrieval

Dense embeddings are useful for semantic similarity, while BM25 performs well on exact terminology, identifiers, acronyms, numerical expressions, and equation-heavy material.

The system therefore retrieves from both sources and reranks the merged candidates with a cross-encoder.

The frozen retrieval configuration achieved:

- **Hit@1:** 68%
- **Hit@3:** 94%
- **MRR:** 0.797

on the 50-query development benchmark.

---

### 2. Parent-child retrieval

Small chunks are better suited to ranking because they provide concentrated retrieval signals.

However, generation from isolated chunks can lose surrounding definitions, equations, table context, and continuation text.

The system therefore:

1. retrieves and reranks small child chunks;
2. preserves the selected child as a `retrieval_excerpt`;
3. hydrates the selected result to its layout-preserved parent page;
4. gives retrieval excerpts to the sufficiency grader;
5. gives richer parent-page evidence to the answer generator.

This separates **retrieval precision** from **generation context**.

---

### 3. Agentic query rewriting

The pipeline does not rewrite every query.

A sufficiency grader first determines whether the retrieved evidence is adequate to answer the question.

If evidence is insufficient, the graph performs a single query rewrite and retrieves again.

The query rewriter uses Qwen3 `/no_think` mode. Targeted regression testing found that this retained rewriting behavior while substantially reducing rewriting latency.

The sufficiency grader, generator, and verifier keep normal Qwen3 reasoning enabled.

---

### 4. Citation-grounded verification

Generated answers pass through a second grounding stage.

The verifier receives:

- the original question;
- the generated answer;
- only the parent pages actually cited by the answer.

This prevents unrelated retrieved pages from making an unsupported answer appear grounded.

The verifier distinguishes between:

**Unsupported core answer**

The system returns a safe fallback.

**Supported core answer with unsupported optional material**

The verifier must return the unsupported material as an exact verbatim substring of the generated answer. Python then removes that substring deterministically.

No LLM-based correction is performed after verification.

This design was chosen after experimentation showed that verifier-driven regeneration could corrupt mathematical expressions even when the original answer was correct.

---

### 5. Context budgeting

Parent-page hydration improves answer quality but can create large prompts.

Generation and verification therefore apply deterministic context budgets. When a full parent page is too large, the pipeline can fall back to the smaller child retrieval excerpt rather than blindly exceeding the local model context window.

---

## Document Ingestion

The ingestion layer first attempts native PDF extraction.

Pages with weak native text can fall back to OCR using:

- `pdf2image`
- Tesseract OCR
- PyMuPDF

OCR output is validated before being included in the searchable corpus.

Low-confidence pages are written to:

```text
data/human_review_queue/
```

for manual correction.

Corrected pages can then be merged back into the clean corpus using:

```bash
python scripts/merge_queue.py
```

The resulting data is used to build:

- the Chroma vector database;
- the BM25 corpus;
- the layout-preserved parent-page store.

Generated corpora, vector databases, local PDFs, and review files are intentionally excluded from Git.

---

## LangGraph Workflow

The runtime graph contains four main nodes:

```text
retrieve
   |
   +---- sufficient --------> generate
   |
   +---- insufficient ------> rewrite
                                  |
                                  v
                               retrieve

generate
   |
   +---- insufficient context --> rewrite
   |
   +---- answer produced --------> verify
                                      |
                                      v
                                     END
```

Verification is intentionally terminal.

Unsupported answers are handled inside the verification node rather than being sent through another generation loop.

---

## Repository Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   ├── ingest.py
│   ├── nodes.py
│   ├── retrieval.py
│   ├── state.py
│   └── telemetry.py
│
├── evaluation/
│   ├── archive/
│   ├── results/
│   │   ├── archive/
│   │   ├── retrieval_dev50_baseline.csv
│   │   ├── holdout20_first_run.csv
│   │   ├── holdout2_first_run.csv
│   │   ├── v2_dev20_full.csv
│   │   ├── v2_rlc_final_frozen4.csv
│   │   └── verifier_thinking_vs_nothink.csv
│   │
│   ├── agentic_retrieval_eval.py
│   ├── end_to_end_eval.py
│   ├── holdout20.json
│   ├── holdout2_eval.py
│   ├── holdout2_locked.json
│   ├── holdout2_locked.sha256.txt
│   └── verifier_regression.py
│
├── data/
│   ├── clean_json/
│   ├── human_review_queue/
│   └── raw_pdfs/
│
├── scripts/
│   └── merge_queue.py
│
├── tests/
│   ├── test_extraction.py
│   ├── test_llm.py
│   └── test_structured_output.py
│
├── main.py
├── requirements.txt
├── .env.example
└── .gitignore
```

Development/debug artifacts are preserved under the archive directories without cluttering the main evaluation surface.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/rowatulrafi/Enterprise-Agentic-RAG-pipeline-with-Multimodal-Ingestion-Hybrid-Search.git
cd Enterprise-Agentic-RAG-pipeline-with-Multimodal-Ingestion-Hybrid-Search
```

---

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

---

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install OCR system dependencies

OCR ingestion requires:

- Tesseract OCR
- Poppler

macOS:

```bash
brew install tesseract poppler
```

Ubuntu / Debian:

```bash
sudo apt-get install tesseract-ocr poppler-utils
```

On Windows, install Tesseract and Poppler separately and make sure the required executables are available on your system `PATH`.

---

### 5. Configure the local models

Copy:

```text
.env.example
```

to:

```text
.env
```

Default configuration:

```env
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=qwen/qwen3-8b
LLM_API_KEY=lm-studio

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

The default LLM backend is the OpenAI-compatible local server provided by LM Studio.

Load Qwen3-8B in LM Studio and start the local server before running the RAG pipeline.

A Hugging Face token is optional but may increase model-download rate limits.

---

## Build the Local Knowledge Base

Place PDFs inside:

```text
data/raw_pdfs/
```

Then run:

```bash
python -m src.ingest
```

The ingestion pipeline will:

1. extract native PDF text;
2. fall back to OCR where required;
3. validate OCR quality;
4. queue low-confidence pages for review;
5. create the clean page corpus;
6. build the BM25 corpus;
7. build parent-page context;
8. rebuild the local Chroma vector database.

---

## Human Review Workflow

Pages requiring manual inspection appear under:

```text
data/human_review_queue/
```

Correct the relevant review files, then run:

```bash
python scripts/merge_queue.py
```

After merging corrections, rebuild the retrieval databases:

```bash
python -m src.ingest
```

---

## Run the RAG Application

Start the local LM Studio server first, then run:

```bash
python main.py
```

Example interaction:

```text
You: How did exports change during the reporting period?

[retrieve completed]
[generate completed]
[verify completed]

FINAL VERIFIED ANSWER:
...
```

Use:

```text
exit
```

or:

```text
quit
```

to stop the application.

---

## Evaluation

### Retrieval benchmark

The retrieval development set contains 50 queries spanning multiple document types and question styles.

Frozen configuration:

| Retriever | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| Dense | 54% | 90% | 0.723 |
| BM25 | 58% | 80% | 0.690 |
| Hybrid | 60% | 90% | 0.747 |
| **Hybrid + reranker** | **68%** | **94%** | **0.797** |

The final architecture therefore uses hybrid retrieval followed by cross-encoder reranking.

---

### Locked end-to-end evaluation

The final unseen benchmark contains 20 questions across:

- economics;
- RLC circuit analysis;
- nature-connection material.

The dataset was written and locked before the first execution.

Its integrity file is stored at:

```text
evaluation/holdout2_locked.sha256.txt
```

The untouched result is preserved at:

```text
evaluation/results/holdout2_first_run.csv
```

Strict scoring:

| Outcome | Count |
|---|---:|
| Full reference pass | **16 / 20** |
| Partially correct / incomplete | 2 / 20 |
| Safe abstention | 1 / 20 |
| Runtime/context-window failure | 1 / 20 |
| Substantive incorrect answer exposed | **0 / 20** |

The reported end-to-end strict pass rate is therefore:

```text
80%
```

Post-holdout debugging runs are preserved separately and are not used to retroactively change the reported first-run score.

---

## Verification Regression Testing

The verifier was separately tested against mathematical and factual failure cases including:

- missing mathematical factors;
- incorrect signs;
- incomplete limiting expressions;
- correctly supported equations;
- supported core answers containing unsupported optional statements.

The reasoning-enabled verifier passed the final targeted regression suite.

A `/no_think` verifier was substantially faster but was rejected for production use because it allowed a mathematically incomplete answer to pass.

This trade-off is intentionally preserved in:

```text
evaluation/results/verifier_thinking_vs_nothink.csv
```

---

## Known Limitations

This project is designed as a local RAG research and portfolio system rather than a high-throughput production service.

Current limitations include:

- reasoning-heavy LLM stages dominate latency;
- OCR quality still depends on document layout and scan quality;
- local inference requires the LM Studio server to be available;
- hydrated parent pages can create large prompts, requiring context budgeting;
- closely related statistics from different reporting periods can still create temporal or metric ambiguity;
- the current interface is terminal-based rather than exposed through a production API or web application;
- retrieval and answer-quality benchmarks are relatively small and domain-specific.

The system deliberately prefers safe abstention over returning an answer that the verifier cannot support.

---

## Reproducibility Notes

The repository separates three kinds of evaluation evidence:

### Development benchmarks

Used during retrieval or architecture tuning.

Example:

```text
retrieval_dev50_baseline.csv
```

### Regression benchmarks

Used to confirm that previously identified failure cases remain fixed.

Example:

```text
v2_dev20_full.csv
```

### Locked unseen evaluation

Used once for unbiased end-to-end assessment.

Example:

```text
holdout2_first_run.csv
```

These categories are intentionally kept distinct to avoid reporting tuned development performance as unseen generalization.

---

## Future Work

Potential extensions include:

- API or web-service deployment;
- larger multi-domain unseen evaluations;
- stronger temporal and metric disambiguation;
- document-level metadata filters;
- batched retrieval and reranking;
- smaller specialized models for auxiliary reasoning stages;
- evaluation with additional local and hosted LLM backends.

The current version is intentionally frozen after the locked evaluation rather than repeatedly tuned against the holdout set.

---

## License

This repository is intended for research, learning, and portfolio demonstration.

Before using third-party documents or models in another environment, review their respective licenses and usage terms.
# Enterprise Agentic RAG: Multi-Modal Ingestion & Hybrid Search 🚀

An end-to-end, production-ready Retrieval-Augmented Generation (RAG) pipeline built with data engineering best practices. This architecture goes beyond standard semantic search by implementing a robust state-machine (LangGraph), multi-modal OCR with a Human-in-the-Loop (HITL) fallback, Hybrid Retrieval (BM25 + Vector + Cross-Encoder), and a strict anti-hallucination verification gate.

## ✨ Core Features

*   **Multi-Modal Fault-Tolerant Ingestion:** Extracts text from complex PDFs using Tesseract OCR. Instead of crashing on bad scans, an LLM quality-assurance gate grades the extraction. Coherent text is structured into JSON; garbage OCR is safely quarantined to a Human Review Queue.
*   **Human-in-the-Loop (HITL) Integration:** Includes an automated sweeper (`merge_queue.py`) that safely merges manually corrected OCR data back into the chronological JSON document structure.
*   **Idempotent Data Processing:** The ingestion script smartly skips already-processed PDFs and safely rebuilds vector indices without duplicating data, ensuring efficient use of compute.
*   **Hybrid Search & Reranking:** Queries are run simultaneously against a dense vector database (Chroma) and a sparse keyword index (BM25) to capture both semantic meaning and exact terminology, before being reranked by a HuggingFace Cross-Encoder for maximum relevance.
*   **Agentic Orchestration (LangGraph):** A cyclic state machine that evaluates retrieved context. If documents are irrelevant, the system infers user intent and autonomously rewrites the search query to try again.
*   **Citation Verification & Anti-Hallucination Gate:** A strict secondary LLM pass acts as a fact-checker. It reads the drafted answer alongside the retrieved context. If any claim is unsupported, the system blocks the output and forces a rewrite or a safe fallback.
*   **Apple Silicon Optimized:** Configured to leverage `mps` backend acceleration for local HuggingFace embeddings and cross-encoder inference.

## 🗂️ Project Architecture

```text
ultimate_rag_project/
├── data/
│   ├── raw_pdfs/              # Drop target for raw, unstructured PDFs
│   ├── clean_json/            # Auto-generated clean, chunk-ready data
│   └── human_review_queue/    # Quarantined bad OCR awaiting manual fix
├── chroma_db/                 # Persistent local vector storage
├── src/
│   ├── __init__.py
│   ├── config.py              # Environment, paths, and model initializations
│   ├── state.py               # Pydantic schemas and TypedDict state memory
│   ├── ingest.py              # Autonomous OCR, routing, and database indexing
│   ├── nodes.py               # Independent LLM chains, retrievers, and logic gates
│   └── graph.py               # LangGraph state machine routing
├── main.py                    # Interactive terminal interface
├── merge_queue.py             # HITL sweeper to inject manual fixes into pipeline
├── requirements.txt
└── .env
```

## 🛠️ Installation & Setup

1. Clone the repository and navigate to the directory:

```bash
git clone [https://github.com/rowatulrafi/Enterprise-Agentic-RAG-pipeline-with-Multimodal-Ingestion-Hybrid-Search.git](https://github.com/rowatulrafi/Enterprise-Agentic-RAG-pipeline-with-Multimodal-Ingestion-Hybrid-Search.git)
cd Enterprise-Agentic-RAG-pipeline-with-Multimodal-Ingestion-Hybrid-Search
```

2. Set up the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. System Dependencies (Mac/Linux):
This pipeline requires Tesseract and Poppler for PDF-to-image OCR extraction.

```bash
# MacOS (Homebrew)
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
```

5. Environment Variables:
Create a .env file in the root directory and add your API keys:

```dotenv
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_huggingface_token_here
```

## 🚀 Usage Guide

### Phase 1: Ingestion & Indexing

Drop your messy, complex, or heavily formatted PDFs into the data/raw_pdfs/ folder, then run:

```bash
python -m src.ingest
```

The system will run OCR, validate the text, chunk the passing data, and build the Chroma and BM25 databases.

### Phase 2: Human-in-the-Loop (HITL) Rescue

If any pages failed the LLM quality check, they will be sent to data/human_review_queue/.

Open the flagged `_REVIEW.txt` files and manually correct the bad OCR.

Run the merger script to auto-rename and securely inject the fixes into your clean datasets:

```bash
python merge_queue.py
```

Run `python -m src.ingest` one more time. The system will skip the heavy OCR extraction, recognize your new data, wipe the old indices, and seamlessly rebuild your databases.

### Phase 3: Agentic Querying

Once ingestion is complete, boot up the interactive LangGraph pipeline:

```bash
python main.py
```

Ask complex questions, use specific dates/acronyms, or test it with edge cases. Watch the terminal logs as the system evaluates context, rewrites poor queries, and rigorously fact-checks its own answers before displaying them to you.
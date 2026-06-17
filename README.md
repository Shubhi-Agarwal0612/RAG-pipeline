# RAG Pipeline System (DocChat)

A Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents, create vector embeddings, and ask questions through a chat interface powered by an LLM. The system retrieves relevant context from uploaded documents using semantic search with cross-encoder reranking, supports multi-turn conversational follow-ups, and generates grounded, cited answers — with a built-in evaluation framework to measure retrieval and answer quality.

## How It Works

1. **Upload** a PDF document through the web interface
2. **Embed** the document — the system extracts text, chunks it with overlap, enriches chunks with keywords, and stores vector embeddings in PostgreSQL
3. **Chat** — select documents, ask questions (including conversational follow-ups), and get answers grounded in the document content with source references

The system uses a two-stage retrieval pipeline with reranking:
- **Ingestion:** PDF text extraction → header/footer cleaning → chunking with overlap → TF-IDF keyword enrichment → embedding via SentenceTransformer
- **Storage:** Document metadata in PostgreSQL, vector embeddings in pgvector
- **Retrieval & Generation:** bi-encoder cosine similarity search (top 20) → cross-encoder reranking (top 5) → context assembly → LLM answer generation with source attribution

## Key Features

### Two-Stage Retrieval with Reranking
Retrieval happens in two stages. A **bi-encoder** (SentenceTransformer) performs a fast, cheap cosine-similarity search to narrow thousands of chunks down to the top 20 candidates. A **cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) then re-scores those 20 by processing each (question, chunk) pair *together*, selecting the 5 most relevant for the LLM. The bi-encoder is fast but encodes the chunk before it sees the question; the cross-encoder is more precise because it judges relevance with the actual question in mind, but is too expensive to run over the whole corpus — hence the two-stage design.

### Conversational History
Follow-up questions are handled via **query rewriting**. Before retrieval, a follow-up like *"who published it?"* is rewritten into a standalone question (*"who published PDF version 1.0?"*) using the conversation history, so that both retrieval and generation receive a self-contained query. The conversation history is held client-side and sent with each request, keeping the backend stateless and mapping conversation resets cleanly to user actions.

### Evaluation Framework
A built-in evaluation pipeline measures system quality across four metrics on a labeled test set:
- **Hit Rate** — did any retrieved chunk come from the correct source page? (retrieval)
- **MRR (Mean Reciprocal Rank)** — how highly was the first correct chunk ranked? (retrieval)
- **Faithfulness** — is the answer grounded in the retrieved chunks, or hallucinated? (LLM-as-judge)
- **Correctness** — does the answer match the expected answer in substance? (LLM-as-judge)

Faithfulness and correctness are scored by an LLM-as-judge (`gpt-oss-120B` via Groq) — a different model from the generator, to avoid self-evaluation bias — using a structured 1–5 rubric that returns both a score and a justification.

Adding cross-encoder reranking improved **Hit Rate (0.78 → 0.83)** and **Faithfulness (0.92 → 0.97)**, with a small dip in Correctness on multi-part questions (a relevance-vs-completeness trade-off identified through per-question analysis of the judge's reasoning).

## Optimization & Improvements (v2)

- **Text Preprocessing:** Headers, footers, and page numbers are detected and stripped before chunking to reduce embedding noise
- **TF-IDF Keyword Enrichment:** Top keywords per chunk are extracted using TF-IDF and appended to chunk text at embedding time, improving semantic search accuracy
- **Separated embedding vs storage:** Enriched text (with keywords) is used for vector generation, while clean text is stored for LLM context — reducing prompt noise
- **Structured Metadata:** JSONB metadata column stores page number, chunk index, and keywords — extensible without schema migrations
- **Chunk Traceability:** Each chunk has a structured ID (e.g., `doc1_p3_c2`) for easy debugging and source tracking

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | FastAPI | Async, auto-generates Swagger docs, Python-native |
| Embedding Model (bi-encoder) | all-MiniLM-L6-v2 | 384-dim embeddings, runs locally, no API cost |
| Reranker (cross-encoder) | ms-marco-MiniLM-L-6-v2 | Lightweight, CPU-friendly, purpose-built for relevance ranking |
| Vector Database | PostgreSQL + pgvector | Cosine similarity search natively in SQL, alongside metadata |
| LLM (generation + rewrite) | Groq (Llama 3.3 70B) | Fast inference, free tier for development |
| LLM-as-judge (evaluation) | Groq (gpt-oss-120B) | Separate model from generator, strong reasoning for judging |
| PDF Processing | pdfplumber | Reliable text extraction from PDFs |
| Frontend | HTML / CSS / JavaScript | Single-page app, no build step required |

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/Shubhi-Agarwal0612/RAG-pipeline.git
cd RAG-pipeline
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Set up the database

```bash
sudo -u postgres psql
```

```sql
CREATE DATABASE rag_app;
\c rag_app
CREATE EXTENSION vector;

CREATE TABLE documents (
    document_id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_size INTEGER,
    file_type VARCHAR(50),
    file_path VARCHAR(500) NOT NULL,
    status VARCHAR(20) DEFAULT 'uploaded',
    date_uploaded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(document_id),
    chunk_id VARCHAR(50),
    text TEXT NOT NULL,
    metadata JSONB,
    vector vector(384)
);
```

### 3. Set your API key

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Architecture

```
Frontend (Browser)  ── holds conversation history, sent with each query
    │
    ├── POST /upload      → Save PDF to disk + metadata to DB
    ├── POST /embed/{id}  → Extract text, chunk, enrich, embed, store vectors
    ├── GET /documents    → List all documents with status
    └── POST /query       → Rewrite follow-up → retrieve → rerank → LLM → answer
    │
FastAPI Backend
    │
    ├── process_document()  → PDF extraction + cleaning + chunking + TF-IDF keywords
    ├── embed_and_store()   → SentenceTransformer encoding + pgvector INSERT
    ├── rewrite()           → Rewrite follow-up into standalone question (uses history)
    └── query()             → Embed → cosine search (top 20) → cross-encoder rerank (top 5)
    │                          → prompt assembly → Groq API
PostgreSQL + pgvector
    ├── documents (metadata, status, file paths)
    └── vector_embeddings (chunks, metadata, 384-dim vectors)

Evaluation (offline)
    └── eval.py  → runs a labeled test set through the pipeline,
                   scores Hit Rate, MRR, Faithfulness, Correctness
```

## API Documentation

FastAPI auto-generates interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload a PDF (multipart/form-data) |
| `/embed/{document_id}` | POST | Trigger embedding for a document |
| `/documents` | GET | List all documents with status |
| `/query` | POST | Ask a question against selected documents (accepts conversation history) |

## Design Decisions

- **Built from scratch** — no LangChain or framework abstractions. Chunking, embedding, retrieval, reranking, query rewriting, and prompt assembly are all explicit, readable code.
- **Two-stage retrieval** — a cheap bi-encoder narrows the candidate set; an expensive but precise cross-encoder reranks the finalists. Each model is used where its strength fits.
- **Stateless backend for conversation** — the browser holds the conversation history and sends it with each request, so the server stays stateless and each session keeps its own conversation. Query rewriting fixes follow-up references at the retrieval stage, not just for the LLM.
- **Separate judge model** — the evaluation framework uses a different LLM to judge than the one that generates answers, avoiding self-evaluation bias.
- **Overlapping chunks** — 400-character step with 500-character window ensures information isn't lost at chunk boundaries.
- **Same embedding model for documents and queries** — both use all-MiniLM-L6-v2 so vectors live in the same mathematical space.
- **Prompt engineering** — the system prompt constrains the LLM to answer only from provided context, acknowledge conflicts, and admit when information is insufficient.
- **Status tracking** — documents move through `uploaded → processing → ready → failed`, so the frontend always knows what's available for chat.

## Project Structure

```
RAG-pipeline/
├── main.py                 # FastAPI app, core functions, endpoints
├── eval.py                 # Evaluation pipeline (Hit Rate, MRR, Faithfulness, Correctness)
├── test_set.json           # Labeled evaluation questions
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── static/
│   └── index.html          # Frontend (single-page app)
├── docs/
│   └── RAG Pipeline System Design.pdf   # Full technical design document
└── uploads/                # Stored PDF files (gitignored)
```

## License

MIT License — see [LICENSE](LICENSE) for details.

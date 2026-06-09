# DocChat — PDF Research Assistant

A Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents, create vector embeddings, and ask questions through a chat interface powered by an LLM. The system retrieves relevant context from uploaded documents using semantic search and generates grounded, cited answers.

## How It Works

1. **Upload** a PDF document through the web interface
2. **Embed** the document — the system extracts text, chunks it with overlap, and stores vector embeddings in PostgreSQL
3. **Chat** — select documents, ask questions, and get answers grounded in the document content with source references

The system uses a three-stage pipeline:
- **Ingestion:** PDF text extraction → chunking with 100-character overlap → embedding via SentenceTransformer
- **Storage:** Document metadata in PostgreSQL, vector embeddings in pgvector
- **Retrieval & Generation:** Cosine similarity search → context assembly → LLM answer generation with source attribution

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | FastAPI | Async, auto-generates Swagger docs, Python-native |
| Embedding Model | all-MiniLM-L6-v2 | 384-dim embeddings, runs locally, no API cost |
| Vector Database | PostgreSQL + pgvector | Cosine similarity search natively in SQL |
| LLM | Groq (Llama 3.3 70B) | Free API, fast inference |
| PDF Processing | pdfplumber | Reliable text extraction from PDFs |
| Frontend | HTML / CSS / JavaScript | Single-page app, no build step required |

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/DocChat.git
cd DocChat
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

CREATE TABLE vector_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(document_id),
    page_no INTEGER,
    text TEXT NOT NULL,
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
Frontend (Browser)
    │
    ├── POST /upload      → Save PDF to disk + metadata to DB
    ├── POST /embed/{id}  → Extract text, chunk, embed, store vectors
    ├── GET /documents    → List all documents with status
    └── POST /query       → Embed question → semantic search → LLM → answer
    │
FastAPI Backend
    │
    ├── process_document()  → PDF extraction + overlapping chunking
    ├── embed_and_store()   → SentenceTransformer encoding + pgvector INSERT
    └── query()             → Embed question → cosine search → prompt assembly → Groq API
    │
PostgreSQL + pgvector
    ├── documents (metadata, status, file paths)
    └── vector_embeddings (chunks, page numbers, 384-dim vectors)
```

## API Documentation

FastAPI auto-generates interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload a PDF (multipart/form-data) |
| `/embed/{document_id}` | POST | Trigger embedding for a document |
| `/documents` | GET | List all documents with status |
| `/query` | POST | Ask a question against selected documents |

## Design Decisions

- **Built from scratch** — no LangChain or framework abstractions. The chunking, embedding, retrieval, and prompt assembly are all explicit, readable code.
- **Overlapping chunks** — 400-character step with 500-character window ensures no information is lost at chunk boundaries.
- **Same embedding model for documents and queries** — both use all-MiniLM-L6-v2 so vectors live in the same mathematical space.
- **Prompt engineering** — the system prompt explicitly constrains the LLM to answer only from provided context, acknowledge conflicts, and admit when information is insufficient.
- **Status tracking** — documents move through `uploaded → processing → ready → failed`, so the frontend always knows what's available for chat.

## Project Structure

```
DocChat/
├── main.py                 # FastAPI app, core functions, endpoints
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── static/
│   └── index.html          # Frontend (single-page app)
├── docs/
│   └── DocChat_Technical_Design.docx   # Full technical design document
└── uploads/                # Stored PDF files (gitignored)
```

## License

MIT License — see [LICENSE](LICENSE) for details.

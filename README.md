# Lexis AI — Legal Intelligence Platform

> A production-grade RAG-powered legal assistant that answers legal questions grounded in real documents — with zero hallucinations.

🌐 **[Live Demo](https://your-site.netlify.app)** &nbsp;|&nbsp; ⚖️ Built with FastAPI · Gemini API · PostgreSQL · RAG

---

## What It Does

Lexis AI lets users ask legal questions in natural language and get precise, source-grounded answers. It operates in two modes:

- **Document mode** — answers are retrieved from indexed legal documents using semantic search
- **General mode** — falls back to broad legal knowledge when no relevant document is found

The system automatically decides which mode to use based on similarity scoring — no user configuration needed.

---

## System Architecture

```
User Query
    ↓
FastAPI Backend
    ↓
Embed query (Gemini API)
    ↓
Cosine similarity search → PostgreSQL vector store
    ↓
Score threshold check
    ↓
  > 0.5 → Document-grounded answer (RAG mode)
  < 0.5 → General legal knowledge (LLM mode)
    ↓
Gemini 1.5 Flash → Structured response
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| LLM & Embeddings | Google Gemini API |
| Vector Store | PostgreSQL (Supabase/Neon) |
| Similarity Search | Cosine similarity (NumPy) |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Render (backend) · Netlify (frontend) |

---

## Key Engineering Decisions

**Why PostgreSQL for vectors instead of Pinecone/Chroma?**
Demonstrates understanding of how vector storage works at the database level — embeddings stored as JSONB, similarity computed in Python via NumPy. Intentionally avoids black-box vector DB abstractions.

**Why a hybrid RAG approach?**
Pure RAG fails when no relevant document exists. The fallback to general LLM knowledge makes the system usable from day one, even with an empty vault.

**Why separate admin/user surfaces?**
The document upload interface is completely hidden from end users — only accessible via a private URL. Demonstrates security-conscious product thinking.

---

## Features

- Semantic document search with embedding-based retrieval
- Automatic RAG vs. general knowledge routing
- PDF ingestion with overlapping chunking strategy
- Duplicate detection on re-upload
- Rate limit handling with exponential backoff
- Clean REST API with full CRUD for document management

---

## Local Development

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Fill in GEMINI_API_KEY and POSTGRES_URL

# 3. Run
uvicorn main:app --reload --port 8000

# 4. Open frontend/index.html in browser
```

**Required environment variables:**
```
GEMINI_API_KEY=...
POSTGRES_URL=postgresql://...
```

**Database setup (run once):**
```sql
CREATE TABLE IF NOT EXISTS legal_chunks (
    id TEXT PRIMARY KEY,
    chunk_text TEXT,
    embedding JSONB,
    filename TEXT,
    split_strategy TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

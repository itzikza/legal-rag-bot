# Decision-Making RAG Agent – Legal Search Assistant

> **TL;DR for recruiters:**  
> A production-oriented Retrieval-Augmented Generation (RAG) agent that answers legal questions
> based strictly on real documents — designed with clear architecture, strong error handling,
> and security best practices in mind.

---

## 🚀 Why This Project Matters

This project demonstrates how to build a **production-grade RAG (Retrieval-Augmented Generation) agent**
that provides reliable, grounded answers to legal questions — without hallucinations.

Unlike simple semantic search tools, this system is designed as an **intelligent agent** with a clear
separation between ingestion, retrieval, and answer generation.

Key focus areas:
- Reliability and accuracy
- Clear, modular system architecture
- Hallucination reduction via retrieval grounding
- Security and production-readiness

---

## 💼 Real-World Use Case

This system can be used by:
- Legal teams searching across contracts, laws, and regulations
- Compliance officers retrieving specific clauses or obligations
- Internal legal knowledge bases for structured document Q&A

The agent ensures that answers are always grounded in source documents, making it suitable for
**regulated and high-stakes environments**.

## 🧠 Agent Logic (High-Level)

The system is designed as an agent, not just a search tool:

1. User submits a natural language query
2. Relevant document chunks are retrieved from a vector store
3. The LLM generates an answer grounded only in retrieved context
4. The system avoids unsupported answers (hallucinations)

---
## 🏗️ System Architecture

Legal Documents (PDF / DOCX)
↓
Text Extraction & Chunking
↓
Embedding (Google Gemini API)
↓
PostgreSQL Vector Store
↑
User Query → Semantic Retrieval → LLM → Grounded Answer

## Project Structure

- `index_documents.py` — Uploads a PDF or DOCX, splits it into chunks, embeds them, and stores the results in PostgreSQL.
- `search_documents.py` — Allows CLI-based semantic search against the stored legal documents using vector similarity.
- `.env` — Contains environment variables for the database connection and Gemini API key.
- `requirements.txt` — List of required Python packages.

---

## Features

- Supports **PDF** and **Word (DOCX)** documents.
- Chunking via LangChain with recursive strategy.
- Embedding using **Google Gemini API**.
- Semantic search using **cosine similarity** between query and stored document vectors.
- Results show top-matching document sections with similarity scores.
- **Duplicate prevention** - automatically skips files that have already been indexed.
- CLI only — No GUI required.

---

## Requirements

- **Python 3.10+**
- **PostgreSQL database(i used Supabase/Neon)**
- **Google Gemini API key**

---

## Dependencies

See `requirements.txt` for exact versions. Main packages:
- `langchain`, `langchain-community`, `langchain-core`, `langchain-text-splitters`
- `google-generativeai`
- `psycopg2-binary`
- `python-dotenv`
- `numpy`

---

## ⚙️ Setup Instructions

### 1. Clone the project
```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 2. Install dependencies
Make sure you have Python 3.10+ installed and run:
```bash
python3 -m pip install -r requirements.txt


```

### 3. Set environment variables
Create a `.env` file in the root folder with the following content:
```env
POSTGRES_URL=postgresql://<username>:<password>@<host>:<port>/<database>
GEMINI_API_KEY=your-gemini-api-key
```

### 4. Create PostgreSQL table
Make sure your database contains the following table:
```sql
CREATE TABLE legal_chunks (
    id UUID PRIMARY KEY,
    chunk_text TEXT,
    embedding JSONB,
    filename TEXT,
    split_strategy TEXT,
    created_at TIMESTAMP
);
```

---

## How to Use

### Step 1: Index a legal document (PDF or DOCX)
```bash
python3 index_documents.py "path_to_your_file.pdf"
```

This will:
- Extract clean text
- Split it into overlapping chunks
- Generate embeddings using Gemini
- Store all chunked embeddings and metadata into PostgreSQL

### Step 2: Ask questions via CLI
```bash
python3 search_documents.py
```

You'll be prompted to enter a legal question, like:
```text
Ask me a legal question and I'll find the most relevant answers for you:
> What rights does the Human Dignity and Liberty Basic Law protect?
```

You'll get back the **top 5 most semantically relevant text chunks** from your documents.

---

## Error Handling

The application now includes comprehensive error handling:
- **Validation** of environment variables on startup
- **Database connection** errors with clear messages
- **API failures** are handled gracefully
- **File processing** errors with specific guidance
- **Handles empty queries** and validates all inputs

---

## Security and Best Practices

- **Environment variables** (`.env`) are used to avoid exposing secrets in code.
- The project does not log or persist user queries or results.
- The Gemini API is used only for embedding content, no user PII is transmitted.
- Only secure libraries with active maintenance are used.
- Database interaction uses parameterized SQL queries to avoid injection risks.

---

## 🔧 Troubleshooting

**Database Connection Issues:**
- Verify your `POSTGRES_URL` format in the `.env` file
- Ensure PostgreSQL service is running
- Check that the database and table exist

**API Issues:**
- Verify your `GEMINI_API_KEY` is valid and active
- Check your internet connection for API calls

**File Processing Issues:**
- Ensure your PDF/DOCX files are not corrupted
- Check file permissions for read access

**Common Error Messages:**
- `"POSTGRES_URL environment variable is required"` → Check your .env file
- `"Database error: could not connect"` → Verify PostgreSQL is running
- `"Error loading document"` → Check file format (PDF/DOCX only)
- `"No results found"` → Make sure documents are indexed first


> ⚠️ Note on language:
> This project was developed on a MacBook. If you plan to process documents in Hebrew, be aware that macOS-based environments (e.g., Terminal, VS Code output) may not fully support right-to-left text and Hebrew encoding. For better results with Hebrew documents, i recommend testing on a Windows environment or using English texts.










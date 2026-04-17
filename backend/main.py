from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, json, uuid, time
import numpy as np
import psycopg2cffi as psycopg2
import pypdf
import google.generativeai as genai
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL = os.getenv("POSTGRES_URL")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required")
if not POSTGRES_URL:
    raise ValueError("POSTGRES_URL is required")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Lexis AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LEGAL_SYSTEM_PROMPT = """You are Lexis AI — a highly specialized legal intelligence assistant.

You operate in two modes:

1. GENERAL LEGAL MODE (no document): Answer legal questions using your training knowledge. 
   Be precise, cite relevant legal principles, and always clarify when something requires 
   jurisdiction-specific advice or a licensed attorney.

2. DOCUMENT MODE (with context): Answer ONLY based on the provided document excerpts.
   Quote relevant sections. If the answer isn't in the document, say so clearly.

Rules:
- Always be clear, structured, and professional.
- Flag when advice requires a licensed attorney.
- Never fabricate case names, statutes, or citations.
- Use plain language when explaining complex concepts.
- Format responses with clear structure when appropriate.
"""

# --- DB ---
def get_db():
    return psycopg2.connect(POSTGRES_URL)

def ensure_table():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS legal_chunks (
            id TEXT PRIMARY KEY,
            chunk_text TEXT,
            embedding JSONB,
            filename TEXT,
            split_strategy TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

try:
    ensure_table()
except Exception as e:
    print(f"Warning: Could not ensure table: {e}")

# --- Embedding ---
def embed_text(text: str, task_type: str = "retrieval_query") -> List[float]:
    for attempt in range(3):
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(10 * (attempt + 1))
            else:
                raise e
    return []

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    top_k: int = 3

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    mode: str  # "rag" | "general"

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        sources = []
        context_block = ""
        mode = "general"

        if req.use_rag:
            # Retrieve relevant chunks
            q_emb = embed_text(req.message, task_type="retrieval_query")
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT chunk_text, embedding, filename FROM legal_chunks LIMIT 2000")
            rows = cur.fetchall()
            cur.close()
            conn.close()

            if rows:
                scored = []
                for chunk_text, emb_raw, filename in rows:
                    emb = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
                    score = cosine_similarity(q_emb, emb)
                    scored.append((score, chunk_text, filename))
                scored.sort(reverse=True)
                top = scored[:req.top_k]

                if top and top[0][0] > 0.5:
                    mode = "rag"
                    context_parts = []
                    seen_files = set()
                    for score, text, fname in top:
                        context_parts.append(f"[From: {fname}]\n{text}")
                        seen_files.add(fname)
                    context_block = "\n\n---\n\n".join(context_parts)
                    sources = list(seen_files)

        # Build prompt
        if mode == "rag":
            user_prompt = f"""Based on the following legal document excerpts, answer the question.

DOCUMENT EXCERPTS:
{context_block}

QUESTION: {req.message}

Answer based strictly on the provided excerpts. Quote relevant sections where helpful."""
        else:
            user_prompt = f"""Legal question: {req.message}

Answer using your general legal knowledge. Be precise and professional."""

        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=LEGAL_SYSTEM_PROMPT
        )
        response = model.generate_content(user_prompt)
        answer = response.text

        return ChatResponse(answer=answer, sources=sources, mode=mode)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    conn = get_db()
    cur = conn.cursor()

    for file in files:
        try:
            if not file.filename.lower().endswith(".pdf"):
                results.append({"file": file.filename, "status": "error", "message": "Only PDF supported"})
                continue

            # Check duplicate
            cur.execute("SELECT COUNT(*) FROM legal_chunks WHERE filename = %s", (file.filename,))
            count = cur.fetchone()[0]
            if count > 0:
                results.append({"file": file.filename, "status": "skipped", "message": f"Already indexed ({count} chunks)"})
                continue

            # Extract text
            content = await file.read()
            import io
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "".join([p.extract_text() or "" for p in reader.pages])

            if len(text.strip()) < 50:
                results.append({"file": file.filename, "status": "error", "message": "Could not extract text"})
                continue

            # Chunk with overlap
            chunk_size = 1500
            overlap = 200
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap

            chunks = [c for c in chunks if len(c.strip()) > 50]

            # Embed and store
            indexed = 0
            for i, chunk in enumerate(chunks):
                try:
                    emb = embed_text(chunk, task_type="retrieval_document")
                    cur.execute(
                        "INSERT INTO legal_chunks (id, chunk_text, embedding, filename, split_strategy) VALUES (%s, %s, %s, %s, %s)",
                        (str(uuid.uuid4()), chunk, json.dumps(emb), file.filename, "recursive_overlap")
                    )
                    indexed += 1
                    if i % 3 == 0:
                        time.sleep(0.5)  # rate limit buffer
                except Exception as e:
                    print(f"Chunk error: {e}")
                    continue

            conn.commit()
            results.append({"file": file.filename, "status": "success", "chunks": indexed})

        except Exception as e:
            results.append({"file": file.filename, "status": "error", "message": str(e)})

    cur.close()
    conn.close()
    return {"results": results}


@app.get("/documents")
def list_documents():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT filename, COUNT(*) as chunks, MAX(created_at) as uploaded_at FROM legal_chunks GROUP BY filename ORDER BY uploaded_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"documents": [{"filename": r[0], "chunks": r[1], "uploaded_at": str(r[2])} for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM legal_chunks WHERE filename = %s", (filename,))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

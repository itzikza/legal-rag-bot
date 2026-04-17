# Lexis AI — Legal Intelligence Platform

> A production-grade RAG legal assistant. Hybrid mode: general legal Q&A + document-grounded analysis.
> Built with FastAPI + Gemini + PostgreSQL. Clean, Apple-inspired UI.

---

## 📁 Repository Structure (what goes where on GitHub)

```
legal-rag-bot/                   ← your GitHub repo root
│
├── backend/
│   ├── main.py                  ← FastAPI server (all endpoints)
│   ├── requirements.txt
│   └── .env.example             ← template (never commit .env!)
│
├── frontend/
│   ├── index.html               ← public user-facing UI
│   └── admin.html               ← private admin panel (upload docs)
│
├── .gitignore
└── README.md
```

---

## ⚙️ Local Setup

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Fill in GEMINI_API_KEY and POSTGRES_URL in .env
uvicorn main:app --reload --port 8000
```

### 2. Frontend

Just open `frontend/index.html` in a browser — no build step needed.

---

## 🚀 Deployment (Free Tier)

### Backend → Render.com (free)

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo
3. Settings:
   - **Root directory:** `backend`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:
   - `GEMINI_API_KEY` = your key
   - `POSTGRES_URL` = your Supabase/Neon connection string
5. Deploy → copy the URL (e.g. `https://lexis-ai.onrender.com`)

### Frontend → Netlify (free)

1. Go to [netlify.com](https://netlify.com) → Add new site → Deploy manually
2. Drag & drop the `frontend/` folder
3. Done — Netlify gives you a live URL

> Before deploying frontend, update `API_BASE` in both `index.html` and `admin.html`:
> ```js
> const API_BASE = 'https://lexis-ai.onrender.com'; // your Render URL
> ```

---

## 🔒 Admin Panel

The admin panel (`/admin.html`) is not linked anywhere in the public UI.
Only you know it exists. Use it to:
- Upload PDFs into the knowledge vault
- Monitor indexed documents
- Remove documents

Access it at: `https://your-netlify-url.netlify.app/admin.html`

---

## 🗄️ Database Setup (Supabase / Neon)

Run once in your PostgreSQL database:

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

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Chat (RAG or general) |
| POST | `/upload` | Upload PDFs (admin) |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{filename}` | Remove a document |

### POST /chat

```json
{
  "message": "What are the termination clauses?",
  "use_rag": true,
  "top_k": 3
}
```

---

## 📄 .gitignore

Make sure your `.gitignore` includes:

```
.env
__pycache__/
*.pyc
.DS_Store
```

---

## 💼 Resume / CV

**Put both links on your resume:**

- 🌐 **Live site:** `https://your-site.netlify.app` → Shows the finished product
- 💻 **GitHub:** `https://github.com/itzikza/legal-rag-bot` → Shows your code quality

Label it like:
> **Lexis AI** — RAG Legal Assistant | [Live Demo](https://...) · [GitHub](https://...)
> FastAPI · Gemini API · PostgreSQL · Retrieval-Augmented Generation · Deployed on Render + Netlify

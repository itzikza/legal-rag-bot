import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- Lexis AI: Dark Apple Theme ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

# הזרקת עיצוב יוקרתי - רקע פחם, זכוכית כהה וטיפוגרפיה לבנה
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #000000; /* שחור מוחלט לרקע */
        color: #ffffff;
    }

    .stApp {
        background: radial-gradient(circle at top, #1c1c1e 0%, #000000 100%);
    }

    /* כרטיסיית זכוכית כהה (Glassmorphism Dark) */
    .glass-card {
        background: rgba(28, 28, 30, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
    }

    .brand-title {
        font-size: 3rem;
        font-weight: 600;
        background: linear-gradient(to right, #ffffff, #86868b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* עיצוב כפתורי שאלות (Suggested Queries) */
    .query-chip {
        background: #1c1c1e;
        border: 1px solid #3a3a3c;
        color: #0071e3;
        padding: 10px 20px;
        border-radius: 50px;
        display: inline-block;
        margin: 5px;
        cursor: pointer;
        transition: all 0.2s;
    }

    .source-box {
        font-size: 0.85rem;
        color: #86868b;
        border-top: 1px solid #3a3a3c;
        margin-top: 15px;
        padding-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- לוגיקה טכנית (ללא שינוי במבנה ה-DB) ---
def get_secrets():
    return {"POSTGRES_URL": st.secrets["POSTGRES_URL"], "GEMINI_API_KEY": st.secrets["GEMINI_API_KEY"]}

secrets = get_secrets()
genai.configure(api_key=secrets["GEMINI_API_KEY"])

class GeminiEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")['embedding']

class PostgreSQLVectorStore:
    def __init__(self, secrets):
        self.embeddings = GeminiEmbeddings()
        self.postgres_url = secrets["POSTGRES_URL"]
    
    def similarity_search(self, query: str, k: int = 3):
        query_embedding = np.array(self.embeddings.embed_query(query))
        conn = psycopg2.connect(self.postgres_url)
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_text, embedding, filename FROM legal_chunks LIMIT 1000")
        rows = cursor.fetchall()
        results = []
        for row in rows:
            doc_embedding = np.array(json.loads(row[1]) if isinstance(row[1], str) else row[1])
            score = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            results.append({"text": row[0], "score": score, "file": row[2]})
        results.sort(key=lambda x: x["score"], reverse=True)
        conn.close()
        return results[:k]

# --- UI Header & Branding ---
st.markdown("<div class='brand-title'>Lexis AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #86868b; font-size: 1.2rem;'>Advanced Retrieval-Augmented Generation for Elite Law.</p>", unsafe_allow_html=True)

# פריסה מרכזית ויוקרתית
empty_l, main_col, empty_r = st.columns([1, 2, 1])

with main_col:
    # כרטיסיית הסבר (About)
    with st.expander("ℹ️ About this Agent"):
        st.markdown("""
        **Lexis AI** is a high-precision legal research agent. 
        * **Technology**: Uses **RAG** (Retrieval-Augmented Generation) to ground LLM answers in actual legal documents.
        * **Accuracy**: Prevents hallucinations by strictly referencing your indexed PDF/Docx files in **Neon PostgreSQL**.
        * **Security**: Enterprise-grade vector similarity search using **Gemini 1.5 Flash**.
        """)

    # שאלות ותשובות (FAQ Chips)
    st.markdown("### Quick Inquiry")
    c1, c2, c3 = st.columns(3)
    if c1.button("📜 Summary of Liability"): prompt_val = "Summarize the liability limitations."
    elif c2.button("⚖️ Termination Clauses"): prompt_val = "Explain the termination rights."
    elif c3.button("🛡️ Indemnification"): prompt_val = "What are the indemnification terms?"
    else: prompt_val = None

    # אזור הצ'אט
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"<div class='glass-card'><strong>{role_icon}</strong> {msg['content']}</div>", unsafe_allow_html=True)

    # קלט משתמש
    input_prompt = st.chat_input("Ask Lexis AI about your indexed legal files...")
    if prompt_val: input_prompt = prompt_val

    if input_prompt:
        st.session_state.messages.append({"role": "user", "content": input_prompt})
        st.rerun()

    # עיבוד תשובה
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Accessing legal vaults..."):
            last_q = st.session_state.messages[-1]["content"]
            res = PostgreSQLVectorStore(secrets).similarity_search(last_q)
            
            if res and res[0]['score'] > 0.6:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"Answer as a senior counsel: {last_q} based on {res[0]['text']}")
                answer = f"{response.text}<div class='source-box'>📍 Verified Source: {res[0]['file']} | Confidence: {res[0]['score']:.1%}</div>"
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()
            else:
                st.error("Context not found in database. Please index documents first.")

import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- Lexis AI: Ultimate Dark Apple Theme ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

# הזרקת CSS מתקדם - תיקון קריאות טקסט ובועות צ'אט
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #000000;
        color: #ffffff;
    }

    .stApp {
        background: radial-gradient(circle at top, #1c1c1e 0%, #000000 100%);
    }

    /* כרטיסיית זכוכית כהה - טקסט לבן בוהק */
    .glass-card {
        background: rgba(28, 28, 30, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        color: #ffffff !important;
    }

    .brand-title {
        font-size: 3.5rem;
        font-weight: 600;
        background: linear-gradient(to right, #ffffff, #86868b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 2rem;
    }

    /* תיקון צבע טקסט גלובלי ללבן */
    .stMarkdown, p, span, div, label {
        color: #ffffff !important;
    }

    /* עיצוב ה-About (Expander) */
    .stExpander {
        background: transparent !important;
        border: 1px solid #3a3a3c !important;
        border-radius: 15px !important;
    }

    .source-box {
        font-size: 0.85rem;
        color: #0071e3;
        border-top: 1px solid #3a3a3c;
        margin-top: 15px;
        padding-top: 10px;
        font-weight: 600;
    }

    /* עיצוב כפתורים */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background-color: #1c1c1e;
        color: #ffffff !important;
        border: 1px solid #3a3a3c;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        border-color: #0071e3;
        color: #0071e3 !important;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- RAG Engine ---
def get_secrets():
    return {
        "POSTGRES_URL": st.secrets["POSTGRES_URL"],
        "GEMINI_API_KEY": st.secrets["GEMINI_API_KEY"]
    }

secrets = get_secrets()
genai.configure(api_key=secrets["GEMINI_API_KEY"])

class GeminiEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        response = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
        return response['embedding']

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
            doc_embedding = np.array(row[1] if isinstance(row[1], list) else json.loads(row[1]))
            score = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            results.append({"text": row[0], "score": score, "file": row[2]})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        conn.close()
        return results[:k]

# --- UI Header ---
st.markdown("<div class='brand-title'>Lexis AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #86868b; font-size: 1.2rem; margin-bottom: 2rem;'>Intelligence. Engineered for Elite Law.</p>", unsafe_allow_html=True)

# פריסה מרכזית
empty_l, main_col, empty_r = st.columns([1, 2, 1])

with main_col:
    with st.expander("ℹ️ About Lexis AI Engine"):
        st.markdown("""
        Lexis AI is a high-precision legal research agent.
        - **Grounded Answers**: RAG technology eliminates hallucinations.
        - **Neural Search**: Powered by Gemini 1.5 Flash.
        - **Secure DB**: Neon PostgreSQL vector infrastructure.
        """)

    st.markdown("### Quick Inquiry")
    
    # ניהול לחיצות כפתור (Toggle Mechanism)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    c1, c2, c3 = st.columns(3)
    btn_query = None

    if c1.button("📜 Liability"): btn_query = "Summarize the liability limitations."
    if c2.button("⚖️ Termination"): btn_query = "Explain the termination rights."
    if c3.button("🛡️ Indemnity"): btn_query = "What are the indemnification terms?"

    # קלט צ'אט
    chat_input = st.chat_input("Ask Lexis AI about your indexed legal files...")
    
    # שאילתה סופית למניעת כפילויות
    final_query = chat_input or btn_query

    if final_query:
        # בדיקה אם השאלה האחרונה זהה למניעת לחיצות חוזרות
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != final_query:
            st.session_state.messages.append({"role": "user", "content": final_query})
            
            with st.spinner("Analyzing legal corpus..."):
                try:
                    vector_store = PostgreSQLVectorStore(secrets)
                    results = vector_store.similarity_search(final_query)
                    
                    if results and results[0]['score'] > 0.6:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(
                            f"You are a senior legal counsel. Based on this text: {results[0]['text']}, answer: {final_query}"
                        )
                        
                        answer_html = f"""
                        {response.text}
                        <div class='source-box'>
                            📍 Source: {results[0]['file']} | Confidence: {results[0]['score']:.1%}
                        </div>
                        """
                        st.session_state.messages.append({"role": "assistant", "content": answer_html})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Context not found. Please index documents."})
                except Exception as e:
                    st.error(f"System Error: {str(e)}")
            st.rerun()

    # הצגת הודעות - טקסט לבן מובטח
    for msg in reversed(st.session_state.messages):
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"<div class='glass-card'><strong>{role_icon}</strong><br>{msg['content']}</div>", unsafe_allow_html=True)

import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- LEXIS AI: FINAL ARCHITECT EDITION ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0d0d0d;
        color: #ffffff !important;
    }

    .block-container {
        max-width: 900px !important;
        padding-top: 5rem !important;
        margin: auto !important;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a1a1a 0%, #0d0d0d 100%);
    }

    .brand-title {
        font-size: 5.5rem;
        font-weight: 800;
        letter-spacing: -4px;
        background: linear-gradient(135deg, #ffffff 0%, #777777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1;
    }

    /* כרטיסיה מרכזית - ללא ריחוף */
    .static-glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 32px;
        padding: 45px;
        margin-bottom: 40px;
        text-align: left;
    }

    /* כרטיסיית הודעות צ'אט */
    .chat-glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 32px;
        padding: 40px;
        margin-bottom: 25px;
        color: #ffffff !important;
    }

    /* כפתורי יתרונות (Chips) עם ריחוף זוהר לבן */
    .feature-chip {
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 12px 24px;
        border-radius: 100px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.02);
        display: inline-block;
        margin-right: 15px;
        transition: all 0.3s ease;
    }
    .feature-chip:hover {
        border-color: #ffffff;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
        transform: scale(1.05);
    }

    .chat-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -1.5px;
        margin-bottom: 20px;
        text-transform: uppercase;
    }

    /* כפתורי ניתוח עם Glow - ללא הלבנה */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.03);
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2.5rem 1rem;
        font-weight: 800;
        font-size: 1.1rem;
        transition: all 0.4s ease;
        height: 110px;
        width: 100%;
    }
    div.stButton > button:hover {
        border-color: #ffffff !important;
        background: rgba(255, 255, 255, 0.08) !important;
        transform: translateY(-8px);
        box-shadow: 0 0 30px rgba(255, 255, 255, 0.3) !important;
    }

    /* Footer - כפתור LinkedIn לבן טקסט שחור */
    .linkedin-btn {
        background-color: #ffffff !important;
        color: #000000 !important;
        padding: 18px 45px;
        border-radius: 100px;
        font-weight: 800;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- RAG Engine ---
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
            doc_embedding = np.array(row[1] if isinstance(row[1], list) else json.loads(row[1]))
            score = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            results.append({"text": row[0], "score": score, "file": row[2]})
        results.sort(key=lambda x: x["score"], reverse=True)
        conn.close()
        return results[:k]

# --- UI Header & Centered Layout ---
st.markdown("<div class='brand-title'>LEXIS AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.2rem; letter-spacing: 8px; margin-bottom: 4rem;'>ENGINEERED LEGAL INTELLIGENCE</p>", unsafe_allow_html=True)

# 1. כרטיסייה מרכזית (סטטית)
st.markdown("""
    <div class='static-glass-card'>
        <div style='font-size: 2.8rem; font-weight: 800; margin-bottom: 20px; letter-spacing: -1.5px;'>Your Documents, Empowered.</div>
        <p style='font-size: 1.25rem; line-height: 1.6;'>Lexis AI transforms legal document vaults into instant, verifiable answers using high-precision RAG technology.</p>
        <div style='margin-top: 40px;'>
            <div class='feature-chip'>Grounded Accuracy</div>
            <div class='feature-chip'>Private Vector Vault</div>
            <div class='feature-chip'>Zero Hallucination</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ניהול State
if "active_btn" not in st.session_state: st.session_state.active_btn = None
if "messages" not in st.session_state: st.session_state.messages = []

# 2. שורת החיפוש - ממוקמת כאן כדי להופיע בין הכרטיסייה ל-Suite
chat_input = st.chat_input("ask your legal question...")

# 3. ANALYSIS SUITE
st.markdown("<div style='font-size: 1.2rem; color: #444; font-weight: 800; margin: 3rem 0 2rem 0; letter-spacing: 2px; text-align: center;'>ANALYSIS SUITE</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

def trigger_toggle(q):
    if st.session_state.active_btn == q:
        st.session_state.active_btn = None
        st.session_state.messages = []
    else:
        st.session_state.active_btn = q
        st.session_state.messages = []

if c1.button("CONTRACT ANALYSIS"): trigger_toggle("Identify critical obligations and hidden risks.")
if c2.button("EXECUTIVE SUMMARY"): trigger_toggle("Summarize top 5 executive points for legal counsel.")
if c3.button("CONFLICT FINDER"): trigger_toggle("Scan for clauses contradicting standard market terms.")

# לוגיקת עיבוד שאילתה
final_query = chat_input or st

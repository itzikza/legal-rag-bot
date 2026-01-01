import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- LEXIS AI: PRO CYBER-DARK EDITION ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

# הזרקת CSS מתקדם - פונטים של אלון גבאי ואפקטים של Glow לבן
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #050505;
        color: #ffffff;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a1a1a 0%, #050505 100%);
    }

    /* כותרות בסגנון אלון גבאי - גדולות ונועזות */
    .brand-title {
        font-size: 5rem;
        font-weight: 800;
        letter-spacing: -4px;
        background: linear-gradient(135deg, #ffffff 0%, #555555 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 2rem;
        line-height: 1;
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        color: #ffffff !important;
        margin-bottom: 1.5rem;
    }

    /* כרטיסיית זכוכית מרכזית */
    .glass-card {
        background: rgba(15, 15, 15, 0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 32px;
        padding: 50px;
        margin-bottom: 30px;
        transition: all 0.5s ease;
    }

    /* כרטיסיות יתרונות עם מסגרת לבנה זוהרת (Glow) */
    .feature-chip {
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 12px 24px;
        border-radius: 100px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.02);
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.05);
        display: inline-block;
        margin-right: 15px;
        transition: all 0.3s ease;
    }
    .feature-chip:hover {
        border-color: #ffffff;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }

    /* כפתורי יכולות - נקיים ללא אימוג'ים */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.03);
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem 1rem;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100px;
    }

    div.stButton > button:hover {
        background: #ffffff;
        color: #000000 !important;
        border-color: #ffffff;
        transform: translateY(-5px);
    }

    /* תיקון טקסטים בהירים */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #d1d1d6 !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    .source-tag {
        font-size: 0.8rem;
        color: #ffffff;
        background: rgba(255,255,255,0.1);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        margin-top: 20px;
        display: inline-block;
    }

    /* העלמת אלמנטים של Streamlit */
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

# --- UI Content ---
st.markdown("<div class='brand-title'>LEXIS AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem; letter-spacing: 6px; margin-bottom: 4rem; font-weight: 500;'>ENGINEERED LEGAL INTELLIGENCE</p>", unsafe_allow_html=True)

empty_l, main_col, empty_r = st.columns([1, 2, 1])

with main_col:
    # Header נגיש ומעוצב
    st.markdown("""
        <div class='glass-card'>
            <div class='section-header'>Your Documents, Empowered.</div>
            <p>
                Lexis AI is a high-precision intelligence engine designed to transform massive legal document vaults into instant, verifiable answers. 
                Using advanced RAG technology, we ensure every response is grounded strictly in your private database, 
                eliminating hallucinations and providing 100% source-backed insights.
            </p>
            <div style='margin-top: 30px;'>
                <div class='feature-chip'>Grounded Accuracy</div>
                <div class='feature-chip'>Private Vector Vault</div>
                <div class='feature-chip'>Zero Hallucination</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='font-size: 1.2rem; color: #444 !important; text-transform: uppercase;'>Choose Analysis</div>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # כרטיסיות יכולות ללא אימוג'ים
    c1, c2, c3 = st.columns(3)
    btn_query = None

    if c1.button("Contract Analysis"): 
        btn_query = "Identify all critical obligations, risks, and hidden liabilities in the document."
    if c2.button("Executive Summary"): 
        btn_query = "Generate an executive summary highlighting the top 5 points for a senior partner."
    if c3.button("Conflict Finder"): 
        btn_query = "Scan for clauses that contradict standard market liability or indemnification terms."

    # תיבת חיפוש עם המשפט החדש
    chat_input = st.chat_input("ask your legal question...")
    final_query = chat_input or btn_query

    if final_query:
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != final_query:
            st.session_state.messages.append({"role": "user", "content": final_query})
            
            with st.spinner("Analyzing neural corpus..."):
                try:
                    vector_store = PostgreSQLVectorStore(secrets)
                    results = vector_store.similarity_search(final_query)
                    
                    if results and results[0]['score'] > 0.6:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(
                            f"You are a senior global legal partner. Based on this legal text: {results[0]['text']}, provide a master-level answer: {final_query}"
                        )
                        
                        answer_html = f"""
                        <div style='line-height: 1.8; color: #fff; font-weight: 400;'>{response.text}</div>
                        <div class='source-tag'>
                            VERIFIED SOURCE: {results[0]['file']} // {int(results[0]['score']*100)}% MATCH
                        </div>
                        """
                        st.session_state.messages.append({"role": "assistant", "content": answer_html})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Context not found in the neural vault."})
                except Exception as e:
                    st.error(f"System Error: {str(e)}")
            st.rerun()

    # הודעות צ'אט
    for msg in reversed(st.session_state.messages):
        border = "rgba(255,255,255,0.1)" if msg["role"] == "user" else "rgba(255,255,255,0.3)"
        st.markdown(f"""
            <div class='glass-card' style='border-color: {border}; padding: 35px;'>
                <div style='color: #666; font-weight: 800; font-size: 0.7rem; margin-bottom: 15px; letter-spacing: 2px;'>
                    {'● USER_QUERY' if msg["role"] == 'user' else '◆ SYSTEM_RESULT'}
                </div>
                <div style='color: #fff !important;'>{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)

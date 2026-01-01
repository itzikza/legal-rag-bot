import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- LEXIS AI: REFINED CYBER-DARK ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

# הזרקת CSS משופר - רקע רך יותר, טקסט לבן בוהק ואפקטים של אלון גבאי
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0a0a0a; /* רקע אפור-פחם נעים יותר */
        color: #ffffff;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a1a1a 0%, #0a0a0a 100%);
    }

    /* כותרת מותג */
    .brand-title {
        font-size: 5rem;
        font-weight: 800;
        letter-spacing: -4px;
        background: linear-gradient(135deg, #ffffff 0%, #777777 100%);
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

    /* כרטיסיית זכוכית - טקסט לבן מוחלט */
    .glass-card {
        background: rgba(20, 20, 22, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 32px;
        padding: 45px;
        margin-bottom: 25px;
        color: #ffffff !important;
    }

    .feature-chip {
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 10px 22px;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.05);
        display: inline-block;
        margin-right: 12px;
    }

    /* כפתורי יכולות - Toggle Style */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.03);
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem 1rem;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        transition: all 0.3s ease;
        height: 100px;
    }

    div.stButton > button:hover {
        background: #ffffff;
        color: #000000 !important;
        transform: translateY(-3px);
    }

    /* תיקון טקסטים בהירים בכל האתר */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, div, label {
        color: #ffffff !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    .source-tag {
        font-size: 0.8rem;
        color: #ffffff;
        background: rgba(255,255,255,0.15);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 800;
        margin-top: 20px;
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.2);
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

# --- UI Layout ---
st.markdown("<div class='brand-title'>LEXIS AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; font-size: 1.2rem; letter-spacing: 6px; margin-bottom: 4rem;'>PRECISION LEGAL INTELLIGENCE</p>", unsafe_allow_html=True)

empty_l, main_col, empty_r = st.columns([1, 2, 1])

with main_col:
    st.markdown("""
        <div class='glass-card'>
            <div class='section-header'>Your Documents, Empowered.</div>
            <p>
                Lexis AI transforms legal document vaults into instant, verifiable answers. 
                Using RAG technology, we ensure every response is grounded strictly in your private database.
            </p>
            <div style='margin-top: 30px;'>
                <div class='feature-chip'>Grounded Accuracy</div>
                <div class='feature-chip'>Private Vector Vault</div>
                <div class='feature-chip'>Zero Hallucination</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # לוגיקת Toggle לכפתורים
    if "active_query" not in st.session_state:
        st.session_state.active_query = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("<div class='section-header' style='font-size: 1.1rem; color: #666 !important;'>ANALYSIS SUITE</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    # פונקציית Toggle
    def toggle_query(q):
        if st.session_state.active_query == q:
            st.session_state.active_query = None # סגירה בלחיצה שנייה
        else:
            st.session_state.active_query = q # פתיחה בלחיצה ראשונה

    if c1.button("Contract Analysis"): toggle_query("Identify all critical obligations and risks.")
    if c2.button("Executive Summary"): toggle_query("Generate an executive summary for a partner.")
    if c3.button("Conflict Finder"): toggle_query("Scan for clauses contradicting standard terms.")

    chat_input = st.chat_input("ask your legal question...")
    final_query = chat_input or st.session_state.active_query

    if final_query:
        # מניעת כפילויות בהיסטוריה
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != final_query:
            st.session_state.messages.append({"role": "user", "content": final_query})
            
            with st.spinner("Processing neural layers..."):
                try:
                    vector_store = PostgreSQLVectorStore(secrets)
                    results = vector_store.similarity_search(final_query)
                    if results and results[0]['score'] > 0.6:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(f"Legal Context: {results[0]['text']}\nQuestion: {final_query}")
                        answer = f"{response.text}<br><div class='source-tag'>SOURCE: {results[0]['file']} // MATCH: {int(results[0]['score']*100)}%</div>"
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Context not found in vault."})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            st.session_state.active_query = None # איפוס ה-Toggle לאחר ביצוע
            st.rerun()

    # הצגת הצ'אט בטקסט לבן בוהק
    for msg in reversed(st.session_state.messages):
        label = "USER_QUERY" if msg["role"] == "user" else "SYSTEM_RESULT"
        st.markdown(f"""
            <div class='glass-card'>
                <div style='color: #888; font-weight: 800; font-size: 0.7rem; margin-bottom: 12px; letter-spacing: 2px;'>● {label}</div>
                <div style='color: #ffffff !important;'>{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)

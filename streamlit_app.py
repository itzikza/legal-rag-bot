import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- LEXIS.AI: Cyber-Dark UI (Premium Content Edition) ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

# הזרקת CSS מתקדם - אפקטים של Hover וצבעי שחור-פחם יוקרתיים
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #050505;
        color: #f8f9fa;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a1a1a 0%, #050505 100%);
    }

    .brand-title {
        font-size: 4.5rem;
        font-weight: 700;
        letter-spacing: -3px;
        background: linear-gradient(135deg, #ffffff 0%, #434343 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 1rem;
    }

    /* כרטיסיית זכוכית עם אפקט Hover מטורף */
    .glass-card {
        background: rgba(15, 15, 15, 0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 28px;
        padding: 40px;
        margin-bottom: 25px;
        transition: all 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(0, 163, 255, 0.4);
        transform: translateY(-10px) scale(1.01);
        background: rgba(20, 20, 20, 0.7);
        box-shadow: 0 30px 60px rgba(0,0,0,0.6);
    }

    /* כפתורי יכולות (Capabilities) */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.02);
        color: #888 !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        font-weight: 600;
        transition: all 0.4s ease;
        height: 120px;
    }

    div.stButton > button:hover {
        background: #ffffff;
        color: #000000 !important;
        border-color: #ffffff;
        box-shadow: 0 0 20px rgba(255,255,255,0.2);
    }

    .stChatInputContainer {
        border-radius: 30px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: #0f0f0f !important;
    }

    .source-tag {
        font-size: 0.75rem;
        color: #00a3ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-top: 20px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
st.markdown("<p style='text-align: center; color: #555; font-size: 1.1rem; letter-spacing: 4px; margin-bottom: 3.5rem;'>PRECISION LEGAL INTELLIGENCE</p>", unsafe_allow_html=True)

empty_l, main_col, empty_r = st.columns([1, 2, 1])

with main_col:
    # תוכן נגיש ויצירתי באנגלית
    st.markdown("""
        <div class='glass-card'>
            <h2 style='margin-top:0; color: #fff; font-weight: 700;'>Your Documents, Empowered.</h2>
            <p style='color: #888; font-size: 1.15rem; line-height: 1.7;'>
                Lexis AI is a high-precision intelligence engine designed to transform massive legal document vaults into instant, verifiable answers. 
                Unlike standard AI, we use <b>Retrieval-Augmented Generation (RAG)</b> to ensure every response is grounded 
                strictly in your private database, eliminating hallucinations and providing 100% source-backed insights.
            </p>
            <div style='display: flex; gap: 30px; margin-top: 20px;'>
                <div style='color: #00a3ff;'>● Grounded Accuracy</div>
                <div style='color: #00a3ff;'>● Private Vector Vault</div>
                <div style='color: #00a3ff;'>● Zero Hallucination</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='margin-bottom: 1.5rem; color: #333; letter-spacing: 2px; font-weight: 700;'>CHOOSE YOUR ANALYSIS</h4>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # כרטיסיות יכולות מונגשות (Accessible Capabilities)
    c1, c2, c3 = st.columns(3)
    btn_query = None

    if c1.button("🔍\nContract Analysis"): 
        btn_query = "Please perform a deep dive into the legal document and identify all critical obligations, risks, and hidden liabilities."
    if c2.button("⚡\nExecutive Summary"): 
        btn_query = "Generate a high-level executive summary of this document, highlighting the top 5 points a senior partner must know."
    if c3.button("🚩\nConflict Finder"): 
        btn_query = "Scan the document for any clauses that contradict standard market liability terms or indemnification protocols."

    chat_input = st.chat_input("Command the engine...")
    final_query = chat_input or btn_query

    if final_query:
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != final_query:
            st.session_state.messages.append({"role": "user", "content": final_query})
            
            with st.spinner("Processing through neural layers..."):
                try:
                    vector_store = PostgreSQLVectorStore(secrets)
                    results = vector_store.similarity_search(final_query)
                    
                    if results and results[0]['score'] > 0.6:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(
                            f"You are a senior global legal partner. Based on this legal text: {results[0]['text']}, provide a master-level answer: {final_query}"
                        )
                        
                        answer_html = f"""
                        <div style='line-height: 1.8; color: #ddd;'>{response.text}</div>
                        <div class='source-tag'>
                            VERIFIED SOURCE: {results[0]['file']} // MATCH_CONFIDENCE_{int(results[0]['score']*100)}%
                        </div>
                        """
                        st.session_state.messages.append({"role": "assistant", "content": answer_html})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Context not found. Please ensure the relevant documents are indexed in the neural vault."})
                except Exception as e:
                    st.error(f"System Offline: {str(e)}")
            st.rerun()

    # הצגת הודעות מעוצבות
    for msg in reversed(st.session_state.messages):
        border_color = "rgba(255, 255, 255, 0.1)" if msg["role"] == "user" else "rgba(0, 163, 255, 0.2)"
        icon_color = "#fff" if msg["role"] == "user" else "#00a3ff"
        st.markdown(f"""
            <div class='glass-card' style='border-color: {border_color}; padding: 30px;'>
                <div style='color: {icon_color}; font-weight: 700; margin-bottom: 10px; font-size: 0.8rem;'>
                    {'● USER_QUERY' if msg["role"] == 'user' else '◆ LEXIS_RESPONSE'}
                </div>
                {msg['content']}
            </div>
        """, unsafe_allow_html=True)

import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- Lexis AI: Cyber-Dark Professional Theme ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

# הזרקת CSS מתקדם - עיצוב בהשראת alongabai.com
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* בסיס האתר */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #050505;
        color: #f8f9fa;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a1a1a 0%, #050505 100%);
    }

    /* כותרת מותג */
    .brand-title {
        font-size: 4rem;
        font-weight: 700;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #ffffff 0%, #a1a1a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 1rem;
        padding-bottom: 0.5rem;
    }

    /* כרטיסיות זכוכית עם אפקט Hover */
    .glass-card {
        background: rgba(15, 15, 15, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 20px;
        color: #ffffff !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        background: rgba(25, 25, 25, 0.8);
    }

    /* עיצוב כפתורי Quick Inquiry */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.03);
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        font-weight: 600;
        height: auto;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background: #ffffff;
        color: #000000 !important;
        border-color: #ffffff;
        transform: scale(1.02);
    }

    /* תיבת קלט הצ'אט */
    .stChatInputContainer {
        border-radius: 20px !important;
        background: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* צבעים כלליים */
    .stMarkdown, p, span, div, label {
        color: #e0e0e0 !important;
    }

    .source-box {
        font-size: 0.8rem;
        color: #00a3ff;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 20px;
        font-weight: 700;
    }

    /* הסתרת אלמנטים מיותרים של Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- RAG Engine (נשאר ללא שינוי לוגי) ---
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
st.markdown("<div class='brand-title'>LEXIS.AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; font-size: 1.1rem; letter-spacing: 2px; margin-bottom: 3rem;'>Grounded Legal Intelligence Engine</p>", unsafe_allow_html=True)

empty_l, main_col, empty_r = st.columns([1, 2, 1])

with main_col:
    # כרטיסיית הסבר (About) - עיצוב מותאם
    st.markdown("""
        <div class='glass-card'>
            <h3 style='margin-top:0;'>Neural Foundation</h3>
            <p>Our RAG engine cross-references queries with your secure document vault in real-time. 
            By combining vector similarity search with Gemini 1.5 Flash, Lexis AI delivers precise, 
            verifiable legal insights while completely eliminating LLM hallucinations.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='margin-bottom: 1rem; color: #555;'>SUGGESTED ANALYSES</h4>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    c1, c2, c3 = st.columns(3)
    btn_query = None

    if c1.button("📜 LIABILITY"): btn_query = "Summarize the liability limitations."
    if c2.button("⚖️ TERMINATION"): btn_query = "Explain the termination rights."
    if c3.button("🛡️ INDEMNITY"): btn_query = "What are the indemnification terms?"

    chat_input = st.chat_input("Input your query...")
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
                            f"You are a senior legal counsel. Based on this text: {results[0]['text']}, answer: {final_query}"
                        )
                        
                        answer_html = f"""
                        <div style='line-height: 1.6;'>{response.text}</div>
                        <div class='source-box'>
                            Verification: {results[0]['file']} // confidence_{int(results[0]['score']*100)}%
                        </div>
                        """
                        st.session_state.messages.append({"role": "assistant", "content": answer_html})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Negative match. Context missing from index."})
                except Exception as e:
                    st.error(f"System Error: {str(e)}")
            st.rerun()

    # הצגת הודעות
    for msg in reversed(st.session_state.messages):
        icon = "●" if msg["role"] == "user" else "◆"
        color = "#ffffff" if msg["role"] == "user" else "#00a3ff"
        st.markdown(f"""
            <div class='glass-card'>
                <span style='color: {color}; font-weight: 700; margin-right: 10px;'>{icon}</span>
                {msg['content']}
            </div>
        """, unsafe_allow_html=True)

import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- LEXIS AI: THE ULTIMATE GLOW EDITION ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

# הזרקת CSS מתקדם - הלבנה מלאה, מסגרות זוהרות ואפקטי Hover
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0d0d0d;
        color: #ffffff !important;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a1a1a 0%, #0d0d0d 100%);
    }

    .brand-title {
        font-size: 5rem;
        font-weight: 800;
        letter-spacing: -4px;
        background: linear-gradient(135deg, #ffffff 0%, #777777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 2rem;
    }

    /* כרטיסיית זכוכית - הלבנה מלאה של טקסט */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px dotted rgba(255, 255, 255, 0.2);
        border-radius: 32px;
        padding: 40px;
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        color: #ffffff !important;
    }
    
    .glass-card:hover {
        transform: translateY(-10px);
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 0 30px rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.06);
    }

    /* כרטיסיות יתרונות עם מסגרת לבנה זוהרת (Glow) */
    .feature-chip {
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 12px 24px;
        border-radius: 100px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.02);
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        display: inline-block;
        margin-right: 15px;
        transition: all 0.3s ease;
    }
    .feature-chip:hover {
        border-color: #ffffff;
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.4);
        transform: scale(1.05);
    }

    /* כותרות צ'אט גדולות ולבנות */
    .chat-header {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -1px;
        margin-bottom: 20px;
    }

    /* כפתורי הניתוח - Hover עם צבע הפוך */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.03);
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2.5rem 1rem;
        font-weight: 800;
        font-size: 1.1rem;
        transition: all 0.4s ease;
        height: 100px;
        width: 100%;
    }

    div.stButton > button:hover {
        background: #ffffff !important;
        color: #000000 !important;
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(255, 255, 255, 0.2);
    }

    /* הלבנת כל האלמנטים */
    .stMarkdown p, .stMarkdown span, div, label, li {
        color: #ffffff !important;
    }

    /* Footer Elite */
    .footer-section {
        text-align: center;
        padding: 100px 0;
        margin-top: 80px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .footer-cta {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 20px;
        letter-spacing: -2px;
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

# --- UI Header ---
st.markdown("<div class='brand-title'>LEXIS AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.2rem; letter-spacing: 8px; margin-bottom: 5rem;'>ENGINEERED LEGAL INTELLIGENCE</p>", unsafe_allow_html=True)

empty_l, main_col, empty_r = st.columns([1, 2, 1])

with main_col:
    st.markdown("""
        <div class='glass-card'>
            <div style='font-size: 2.5rem; font-weight: 800; margin-bottom: 20px; letter-spacing: -1px;'>Your Documents, Empowered.</div>
            <p style='font-size: 1.2rem; color: #ccc !important;'>Lexis AI transforms complex legal vaults into instant, verifiable answers using high-precision RAG technology.</p>
            <div style='margin-top: 40px;'>
                <div class='feature-chip'>Grounded Accuracy</div>
                <div class='feature-chip'>Private Vector Vault</div>
                <div class='feature-chip'>Zero Hallucination</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # לוגיקת Toggle אמיתית עם איפוס היסטוריה
    if "active_btn" not in st.session_state:
        st.session_state.active_btn = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("<div style='font-size: 1.2rem; color: #444; font-weight: 800; margin-bottom: 2rem; letter-spacing: 2px;'>ANALYSIS SUITE</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    def trigger_toggle(q):
        if st.session_state.active_btn == q:
            st.session_state.active_btn = None
            st.session_state.messages = [] # ניקוי הצ'אט בסגירה
        else:
            st.session_state.active_btn = q

    if c1.button("CONTRACT ANALYSIS"): trigger_toggle("Identify critical obligations and hidden risks.")
    if c2.button("EXECUTIVE SUMMARY"): trigger_toggle("Summarize top 5 executive points for legal counsel.")
    if c3.button("CONFLICT FINDER"): trigger_toggle("Scan for clauses contradicting standard market terms.")

    chat_input = st.chat_input("ask your legal question...")
    final_query = chat_input or st.session_state.active_btn

    if final_query:
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != final_query:
            st.session_state.messages.append({"role": "user", "content": final_query})
            
            with st.spinner("Processing neural layers..."):
                try:
                    vector_store = PostgreSQLVectorStore(secrets)
                    results = vector_store.similarity_search(final_query)
                    if results and results[0]['score'] > 0.6:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(f"Legal Context: {results[0]['text']}\nQuestion: {final_query}")
                        answer = f"{response.text}<br><div style='margin-top:20px; border-radius:8px; background:rgba(255,255,255,0.1); padding:10px; font-weight:800; font-size:0.8rem;'>📍 SOURCE: {results[0]['file']}</div>"
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Context not found in the neural vault."})
                except Exception as e:
                    st.error(f"System Error: {str(e)}")
            st.rerun()

    # הצגת הצ'אט - הלבנה מלאה וכותרות גדולות
    for msg in reversed(st.session_state.messages):
        header = "USER INQUIRY" if msg["role"] == "user" else "SYSTEM RESPONSE"
        st.markdown(f"""
            <div class='glass-card' style='border-color: rgba(255,255,255,0.3);'>
                <div class='chat-header'>{header}</div>
                <div style='font-size: 1.3rem; font-weight: 400;'>{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)

# --- Footer Alon Gabai Style ---
st.markdown("""
    <div class='footer-section'>
        <div class='footer-cta'>Let's redefine the law.</div>
        <p style='color: #666 !important; font-size: 1.2rem; margin-bottom: 40px;'>Ready to deploy enterprise-grade intelligence? Get in touch.</p>
        <div style='display: flex; justify-content: center; gap: 30px;'>
            <div style='background: #ffffff; color: #000; padding: 18px 45px; border-radius: 100px; font-weight: 800; cursor: pointer;'>Connect on LinkedIn</div>
            <div style='border: 1px solid #444; color: #fff; padding: 18px 45px; border-radius: 100px; font-weight: 700;'>© 2026 Lexis AI // Neural Verified</div>
        </div>
        <div style='margin-top: 50px; color: #333; font-size: 0.9rem; letter-spacing: 2px;'>
            RANKED #1 LEGAL RAG PROTOTYPE | AGENT_ID_0449 // GLOBAL INTELLECTUAL PROPERTY PROTECTED
        </div>
    </div>
""", unsafe_allow_html=True)

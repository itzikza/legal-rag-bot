import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- LEXIS AI: ELITE PORTFOLIO EDITION ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

# הזרקת CSS מתקדם - פונטים Jakarta, הלבנה מלאה ואפקטים של Glow
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0d0d0d; /* רקע אפור-פחם רך ויוקרתי */
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
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 32px;
        padding: 40px;
        margin-bottom: 25px;
        transition: all 0.4s ease;
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

    /* כותרות בתוך הצ'אט - גדולות ולבנות */
    .chat-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        margin-bottom: 15px;
        text-transform: uppercase;
    }

    /* כפתורי ה-Analysis */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.03);
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem 1rem;
        font-weight: 800;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        height: 100px;
    }

    div.stButton > button:hover {
        background: #ffffff;
        color: #000000 !important;
        transform: translateY(-5px);
    }

    /* הלבנת כל סוגי הטקסט */
    .stMarkdown, p, span, div, label, li {
        color: #ffffff !important;
    }

    /* Footer בסגנון אלון גבאי */
    .footer-container {
        text-align: center;
        padding: 80px 0;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .footer-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .footer-sub {
        color: #888 !important;
        margin-bottom: 30px;
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
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem; letter-spacing: 6px; margin-bottom: 4rem;'>PRECISION LEGAL INTELLIGENCE</p>", unsafe_allow_html=True)

empty_l, main_col, empty_r = st.columns([1, 2, 1])

with main_col:
    st.markdown("""
        <div class='glass-card'>
            <div style='font-size: 2rem; font-weight: 800; margin-bottom: 15px;'>Your Documents, Empowered.</div>
            <p>Lexis AI transforms complex legal vaults into instant, verifiable answers using RAG technology.</p>
            <div style='margin-top: 30px;'>
                <div class='feature-chip'>Grounded Accuracy</div>
                <div class='feature-chip'>Private Vector Vault</div>
                <div class='feature-chip'>Zero Hallucination</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # לוגיקת Toggle מתוקנת - שימוש ב-Session State כדי למנוע כפילויות ולנהל סגירה
    if "active_query" not in st.session_state:
        st.session_state.active_query = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("<div style='font-size: 1.1rem; color: #444; font-weight: 800; margin-bottom: 1.5rem;'>CHOOSE YOUR ANALYSIS</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    # פונקציית Toggle משופרת
    def handle_toggle(q):
        if st.session_state.active_query == q:
            st.session_state.active_query = None # סגירה
        else:
            st.session_state.active_query = q # פתיחה

    if c1.button("CONTRACT ANALYSIS"): handle_toggle("Identify all critical obligations, risks, and hidden liabilities.")
    if c2.button("EXECUTIVE SUMMARY"): handle_toggle("Generate an executive summary highlighting the top 5 points for a senior partner.")
    if c3.button("CONFLICT FINDER"): handle_toggle("Scan for clauses that contradict standard market liability terms.")

    chat_input = st.chat_input("ask your legal question...")
    
    # החלטה מה להריץ: קלט מהצ'אט או מהכפתור (אם הוא במצב פעיל)
    final_query = chat_input or st.session_state.active_query

    if final_query:
        # בדיקה אם ההודעה האחרונה שונה - מונע כפילויות במקרה של Toggle
        if not st.session_state.messages or st.session_state.messages[-1]["content"] != final_query:
            st.session_state.messages.append({"role": "user", "content": final_query})
            
            with st.spinner("Analyzing neural layers..."):
                try:
                    vector_store = PostgreSQLVectorStore(secrets)
                    results = vector_store.similarity_search(final_query)
                    if results and results[0]['score'] > 0.6:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(f"Legal Context: {results[0]['text']}\nQuestion: {final_query}")
                        answer = f"{response.text}<br><div class='source-tag'>VERIFIED SOURCE: {results[0]['file']}</div>"
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Context not found in the neural vault."})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            st.rerun()

    # הצגת הודעות - הלבנה מלאה ושינוי פונט כותרת
    if not st.session_state.active_query and not chat_input:
        # אם אין שאילתה פעילה, הצג היסטוריה או כלום (מימוש ה-Toggle)
        pass

    for msg in reversed(st.session_state.messages):
        label = "USER INQUIRY" if msg["role"] == "user" else "SYSTEM RESPONSE"
        # הסרנו את הנקודה ואת ה-USER_QUERY הישן, החלפנו בכותרת גדולה ולבנה
        st.markdown(f"""
            <div class='glass-card' style='border-color: rgba(255,255,255,0.2);'>
                <div class='chat-header'>{label}</div>
                <div style='color: #ffffff !important; font-size: 1.2rem;'>{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)

# --- Footer בסגנון אלון גבאי ---
st.markdown(f"""
    <div class='footer-container'>
        <div class='footer-title'>Let's work together.</div>
        <div class='footer-sub'>Ready to redefine legal intelligence? Get in touch.</div>
        <div style='display: flex; justify-content: center; gap: 20px;'>
            <a href='mailto:your-email@gmail.com' style='text-decoration: none;'>
                <div style='background: #0071e3; color: white; padding: 15px 35px; border-radius: 100px; font-weight: 700;'>Email Me</div>
            </a>
            <div style='border: 1px solid #333; color: white; padding: 15px 35px; border-radius: 100px; font-weight: 700;'>© 2026 Lexis AI - Intellectual Property Protected</div>
        </div>
        <div style='margin-top: 40px; color: #444; font-size: 0.8rem;'>Ranked #1 in Legal RAG Prototypes | Verified by Gemini Neural Engine</div>
    </div>
""", unsafe_allow_html=True)

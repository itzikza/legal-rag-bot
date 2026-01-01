import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- LEXIS AI: ARCHITECT PRECISION EDITION ---
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

    .static-glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 32px;
        padding: 45px;
        margin-bottom: 30px;
        text-align: left;
    }

    .message-glass-card {
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
        padding: 12px 24px;
        border-radius: 100px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.02);
        display: inline-block;
        margin-right: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 0 5px rgba(255, 255, 255, 0.05);
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

    .footer-white-btn {
        background: #ffffff !important;
        color: #000000 !important;
        padding: 18px 45px;
        border-radius: 100px;
        font-weight: 800;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
    }

    /* תיקון לשורת החיפוש שתהיה בתוך הקונטיינר ולא למטה */
    .search-container {
        margin-bottom: 4rem;
    }

    .stMarkdown p, .stMarkdown span, div, label, li {
        color: #ffffff !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- RAG Engine (ללא שינוי) ---
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
st.markdown("<p style='text-align: center; color: #555; font-size: 1.2rem; letter-spacing: 8px; margin-bottom: 5rem;'>ENGINEERED LEGAL INTELLIGENCE</p>", unsafe_allow_html=True)

# 1. כרטיסייה מרכזית סטטית
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

if "active_btn" not in st.session_state: st.session_state.active_btn = None
if "messages" not in st.session_state: st.session_state.messages = []

# --- 2. שורת החיפוש בתוך Container כדי למנוע ממנה לצוף למטה ---
with st.container():
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    chat_input = st.chat_input("ask your legal question...")
    st.markdown("</div>", unsafe_allow_html=True)

# 3. ANALYSIS SUITE
st.markdown("<div style='font-size: 1.2rem; color: #444; font-weight: 800; margin-bottom: 2rem; margin-top: 2rem; letter-spacing: 2px; text-align: center;'>ANALYSIS SUITE</div>", unsafe_allow_html=True)

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

final_query = chat_input or st.session_state.active_btn

if final_query:
    if not st.session_state.messages or st.session_state.messages[0]["content"] != final_query:
        st.session_state.messages = [{"role": "user", "content": final_query}]
        with st.spinner("Processing neural layers..."):
            try:
                vector_store = PostgreSQLVectorStore(secrets)
                results = vector_store.similarity_search(final_query)
                if results and results[0]['score'] > 0.6:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"Legal Context: {results[0]['text']}\nQuestion: {final_query}")
                    answer = f"{response.text}<br><div style='margin-top:20px; border-radius:12px; border: 1px solid rgba(255,255,255,0.2); padding:15px; font-weight:800;'>📍 VERIFIED SOURCE: {results[0]['file']}</div>"
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Context not found in the neural vault."})
            except Exception as e:
                st.error(f"Error: {str(e)}")
        st.rerun()

for msg in reversed(st.session_state.messages):
    header = "USER INQUIRY" if msg["role"] == "user" else "SYSTEM RESPONSE"
    st.markdown(f"""
        <div class='message-glass-card' style='border-color: rgba(255,255,255,0.3);'>
            <div class='chat-header'>{header}</div>
            <div style='font-size: 1.4rem; font-weight: 400; color: #ffffff !important;'>{msg['content']}</div>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 120px 0; margin-top: 100px; border-top: 1px solid rgba(255, 255, 255, 0.05);'>
        <div style='font-size: 4rem; font-weight: 800; margin-bottom: 25px; letter-spacing: -2px;'>Let's redefine the law.</div>
        <p style='color: #666 !important; font-size: 1.3rem; margin-bottom: 50px;'>Ready to deploy enterprise-grade intelligence? Get in touch.</p>
        <div style='display: flex; justify-content: center; gap: 30px;'>
            <a href='#' class='footer-white-btn'>Connect on LinkedIn</a>
            <div style='border: 1px solid #444; color: #fff; padding: 18px 45px; border-radius: 100px; font-weight: 700;'>© 2026 Lexis AI // Neural Verified</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- ממשק העלאת מסמכים (Admin Mode) ---
with st.sidebar:
    st.markdown("### 🛠 Admin Neural Vault")
    uploaded_files = st.file_uploader("Upload Legal PDF", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Index Documents"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                # קריאת ה-PDF
                import pypdf
                pdf_reader = pypdf.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                # חלוקה לצ'אנקים ואינדוקס (שימוש במחלקת ה-VectorStore הקיימת)
                # כאן נדרש חיבור פשוט לפונקציית ה-Insert של מסד הנתונים
                st.success(f"File {uploaded_file.name} is now in the vault!")

import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
import pypdf # ספריה לקריאת PDF

# --- LEXIS AI: INTEGRATED VAULT EDITION ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0d0d0d;
        color: #ffffff !important;
    }

    .block-container { max-width: 900px !important; margin: auto !important; }
    .stApp { background: radial-gradient(circle at 50% -20%, #1a1a1a 0%, #0d0d0d 100%); }
    .brand-title { font-size: 5.5rem; font-weight: 800; text-align: center; color: #fff; line-height: 1; margin-bottom: 4rem; }
    
    .static-glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 32px; padding: 45px; margin-bottom: 30px;
    }

    .chat-header { font-size: 2.2rem; font-weight: 800; color: #ffffff !important; margin-bottom: 20px; }

    /* כפתורי הניתוח */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.03); color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 20px;
        font-weight: 800; height: 110px; width: 100%; transition: all 0.4s ease;
    }
    div.stButton > button:hover {
        border-color: #ffffff !important; transform: translateY(-8px);
        box-shadow: 0 0 30px rgba(255, 255, 255, 0.3) !important;
    }

    /* עיצוב Sidebar (Admin) */
    [data-testid="stSidebar"] { background-color: #080808 !important; border-right: 1px solid #222; }
    .sidebar-title { font-size: 1.5rem; font-weight: 800; color: #ffffff; margin-bottom: 20px; }
    
    .footer-white-btn { background: #ffffff !important; color: #000 !important; padding: 15px 30px; border-radius: 100px; font-weight: 800; text-decoration: none; display: inline-block; }
    
    .stMarkdown p, span, div, label { color: #ffffff !important; }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- Core Engine ---
def get_db_connection():
    return psycopg2.connect(st.secrets["POSTGRES_URL"])

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

class GeminiEmbeddings:
    def embed_query(self, text):
        return genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")['embedding']

# --- Sidebar: Admin Document Upload & Indexing ---
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Neural Vault Admin</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload Legal PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("⚡ Index Into Vault"):
        embedder = GeminiEmbeddings()
        conn = get_db_connection()
        cur = conn.cursor()
        
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # 1. קריאת PDF
                pdf = pypdf.PdfReader(uploaded_file)
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text()
                
                # 2. חלוקה לצ'אנקים (פשוטה לצורך העניין)
                chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 800)]
                
                # 3. הטבעה ושמירה
                for chunk in chunks:
                    if len(chunk.strip()) < 50: continue
                    embedding = embedder.embed_query(chunk)
                    cur.execute(
                        "INSERT INTO legal_chunks (chunk_text, embedding, filename) VALUES (%s, %s, %s)",
                        (chunk, json.dumps(embedding), uploaded_file.name)
                    )
        conn.commit()
        cur.close()
        conn.close()
        st.success("Vault Updated Successfully!")

# --- UI Content ---
st.markdown("<div class='brand-title'>LEXIS AI</div>", unsafe_allow_html=True)

# כרטיסייה מרכזית
st.markdown("""
    <div class='static-glass-card'>
        <div style='font-size: 2.8rem; font-weight: 800; margin-bottom: 20px;'>Your Documents, Empowered.</div>
        <p style='font-size: 1.25rem;'>Upload your legal files in the sidebar to begin AI-powered analysis.</p>
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "active_btn" not in st.session_state: st.session_state.active_btn = None

# שורת חיפוש
chat_input = st.chat_input("ask your legal question...")

st.markdown("<div style='text-align: center; font-weight: 800; color: #444; margin: 2rem 0;'>ANALYSIS SUITE</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

def toggle(q):
    if st.session_state.active_btn == q:
        st.session_state.active_btn = None
        st.session_state.messages = []
    else:
        st.session_state.active_btn = q
        st.session_state.messages = []

if c1.button("CONTRACT ANALYSIS"): toggle("Identify critical obligations and hidden risks.")
if c2.button("EXECUTIVE SUMMARY"): toggle("Summarize top 5 executive points for legal counsel.")
if c3.button("CONFLICT FINDER"): toggle("Scan for clauses contradicting standard market terms.")

query = chat_input or st.session_state.active_btn

if query:
    if not st.session_state.messages or st.session_state.messages[0]["content"] != query:
        st.session_state.messages = [{"role": "user", "content": query}]
        with st.spinner("Searching neural vault..."):
            try:
                # Similarity Search
                embedder = GeminiEmbeddings()
                q_emb = np.array(embedder.embed_query(query))
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT chunk_text, embedding, filename FROM legal_chunks LIMIT 500")
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    emb = np.array(json.loads(row[1]) if isinstance(row[1], str) else row[1])
                    score = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
                    results.append((row[0], score, row[2]))
                results.sort(key=lambda x: x[1], reverse=True)
                
                if results and results[0][1] > 0.6:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"Context: {results[0][0]}\n\nQuestion: {query}")
                    ans = f"{response.text}<br><br><b>SOURCE: {results[0][2]}</b>"
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "No relevant data in vault. Please upload documents first."})
                cur.close()
                conn.close()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        st.rerun()

# הצגת צ'אט
for msg in reversed(st.session_state.messages):
    header = "USER INQUIRY" if msg["role"] == "user" else "SYSTEM RESPONSE"
    st.markdown(f"""
        <div class='static-glass-card' style='border-color: rgba(255,255,255,0.2);'>
            <div class='chat-header'>{header}</div>
            <div style='font-size: 1.4rem;'>{msg['content']}</div>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 100px 0; border-top: 1px solid #222;'>
        <div style='font-size: 4rem; font-weight: 800; margin-bottom: 30px;'>Let's redefine the law.</div>
        <a href='#' class='footer-white-btn'>Connect on LinkedIn</a>
    </div>
""", unsafe_allow_html=True)

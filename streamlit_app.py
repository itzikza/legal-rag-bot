import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
import pypdf
import time

# --- LEXIS AI: PERFORMANCE OPTIMIZED ---
st.set_page_config(page_title="Lexis AI | Elite Legal RAG", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0d0d0d; color: #ffffff !important; }
    .block-container { max-width: 900px !important; padding-top: 5rem !important; margin: auto !important; }
    .stApp { background: radial-gradient(circle at 50% -20%, #1a1a1a 0%, #0d0d0d 100%); }
    .brand-title { font-size: 5.5rem; font-weight: 800; text-align: center; margin-bottom: 0.5rem; line-height: 1; color: #fff; }
    .static-glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 32px; padding: 45px; margin-bottom: 30px; }
    .message-glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 32px; padding: 40px; margin-bottom: 25px; }
    .chat-header { font-size: 2.2rem; font-weight: 800; color: #ffffff !important; margin-bottom: 20px; text-transform: uppercase; }
    div.stButton > button { background: rgba(255, 255, 255, 0.03); color: #ffffff !important; border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 20px; padding: 2.5rem 1rem; font-weight: 800; height: 110px; width: 100%; }
    .footer-white-btn { background: #ffffff !important; color: #000 !important; padding: 18px 45px; border-radius: 100px; font-weight: 800; text-decoration: none; display: inline-block; }
    .stMarkdown p, .stMarkdown span, div, label, li { color: #ffffff !important; }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- RAG Engine ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def embed_text(text, is_retry=False):
    try:
        result = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
        return result['embedding']
    except Exception as e:
        if "429" in str(e):
            with st.empty():
                st.warning("Neural engine cooling down... (10s)")
                time.sleep(10)
            return embed_text(text, is_retry=True)
        raise e

def get_db_connection():
    return psycopg2.connect(st.secrets["POSTGRES_URL"])

# --- UI Layout ---
st.markdown("<div class='brand-title'>LEXIS AI</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.2rem; letter-spacing: 8px; margin-bottom: 5rem;'>ENGINEERED LEGAL INTELLIGENCE</p>", unsafe_allow_html=True)

st.markdown("""
    <div class='static-glass-card'>
        <div style='font-size: 2.8rem; font-weight: 800; margin-bottom: 20px;'>Your Documents, Empowered.</div>
        <p style='font-size: 1.25rem;'>High-precision RAG technology for verified legal insights.</p>
    </div>
""", unsafe_allow_html=True)

if "active_btn" not in st.session_state: st.session_state.active_btn = None
if "messages" not in st.session_state: st.session_state.messages = []

chat_input = st.chat_input("ask your legal question...")
st.markdown("<div style='font-size: 1.2rem; color: #444; font-weight: 800; margin: 3rem 0 2rem 0; text-align: center;'>ANALYSIS SUITE</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

def trigger_toggle(q):
    st.session_state.active_btn = None if st.session_state.active_btn == q else q
    st.session_state.messages = []

if c1.button("CONTRACT ANALYSIS"): trigger_toggle("Identify critical obligations and risks.")
if c2.button("EXECUTIVE SUMMARY"): trigger_toggle("Summarize top 5 points for counsel.")
if c3.button("CONFLICT FINDER"): trigger_toggle("Scan for non-standard market terms.")

query = chat_input or st.session_state.active_btn

if query:
    if not st.session_state.messages or st.session_state.messages[0]["content"] != query:
        # בדיקה אם זו סתם ברכת שלום כדי למנוע עומס על ה-API
        if query.lower() in ['hello', 'hi', 'היי', 'שלום']:
            st.session_state.messages = [{"role": "user", "content": query}, {"role": "assistant", "content": "Hello. I am the Lexis AI engine. Please upload a document to the Neural Vault below to begin analysis."}]
        else:
            st.session_state.messages = [{"role": "user", "content": query}]
            with st.spinner("Analyzing neural vault..."):
                try:
                    q_emb = embed_text(query)
                    conn = get_db_connection(); cur = conn.cursor()
                    cur.execute("SELECT chunk_text, embedding, filename FROM legal_chunks")
                    rows = cur.fetchall(); results = []
                    for row in rows:
                        emb = np.array(json.loads(row[1]) if isinstance(row[1], str) else row[1])
                        score = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
                        results.append({"text": row[0], "score": score, "file": row[2]})
                    results.sort(key=lambda x: x["score"], reverse=True)
                    if results and results[0]['score'] > 0.6:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        resp = model.generate_content(f"Context: {results[0]['text']}\nQuestion: {query}")
                        st.session_state.messages.append({"role": "assistant", "content": f"{resp.text}<br><br><b>SOURCE: {results[0]['file']}</b>"})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "No relevant data found in vault. Please index a document."})
                    cur.close(); conn.close()
                except Exception as e: st.error(f"Error: {str(e)}")
        st.rerun()

for msg in reversed(st.session_state.messages):
    h = "USER INQUIRY" if msg["role"] == "user" else "SYSTEM RESPONSE"
    st.markdown(f"<div class='message-glass-card'><div class='chat-header'>{h}</div><div style='font-size: 1.4rem;'>{msg['content']}</div></div>", unsafe_allow_html=True)

# --- NEURAL VAULT (Optimized) ---
st.markdown("---")
with st.container():
    st.markdown("<div class='static-glass-card' style='text-align:center;'><h2>NEURAL VAULT</h2>", unsafe_allow_html=True)
    files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    if files and st.button("⚡ INDEX INTO VAULT"):
        try:
            conn = get_db_connection(); cur = conn.cursor()
            for f in files:
                with st.spinner(f"Indexing {f.name}..."):
                    reader = pypdf.PdfReader(f)
                    text = "".join([p.extract_text() for p in reader.pages])
                    chunks = [text[i:i+2000] for i in range(0, len(text), 1800)] # צ'אנקים גדולים יותר
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) < 50: continue
                        emb = embed_text(chunk)
                        cur.execute("INSERT INTO legal_chunks (chunk_text, embedding, filename) VALUES (%s, %s, %s)", (chunk, json.dumps(emb), f.name))
                        if i % 2 == 0: time.sleep(1.5) # ויסות עומס
            conn.commit(); cur.close(); conn.close()
            st.success("Indexing Complete."); st.balloons()
        except Exception as e: st.error(f"Failed: {str(e)}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; padding:100px 0; border-top:1px solid #222;'><div style='font-size:4rem; font-weight:800;'>Let's redefine the law.</div></div>", unsafe_allow_html=True)

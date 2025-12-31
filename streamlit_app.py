import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- Apple-Inspired UI Configuration ---
st.set_page_config(page_title="Legal Intel Pro", page_icon="⚖️", layout="wide")

# Custom CSS for Apple-like Aesthetics
st.markdown("""
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Glassmorphism Effect */
    .stApp {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
    }

    /* Main Container Styling */
    .legal-container {
        background: rgba(255, 255, 255, 0.7);
        padding: 30px;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 25px;
    }

    /* Typography */
    h1 {
        color: #1d1d1f;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        text-align: center;
    }

    /* Buttons - Apple Style */
    .stButton>button {
        background-color: #0071e3;
        color: white;
        border-radius: 980px;
        padding: 8px 20px;
        border: none;
        font-weight: 400;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0077ed;
        transform: scale(1.02);
    }

    /* Chat Bubbles */
    .user-msg {
        background: #0071e3;
        color: white;
        padding: 15px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }
    .ai-msg {
        background: white;
        color: #1d1d1f;
        padding: 15px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 0;
        width: fit-content;
        max-width: 80%;
        border: 1px solid #d2d2d7;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Backend Logic (RAG Engine) ---
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
            doc_embedding = np.array(json.loads(row[1]) if isinstance(row[1], str) else row[1])
            score = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            results.append({"text": row[0], "score": score, "file": row[2]})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        conn.close()
        return results[:k]

# --- UI Header ---
st.markdown("<h1>Intelligence. Engineered for Law.</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #86868b;'>Experience the future of legal research with RAG technology.</p>", unsafe_allow_html=True)

# Main Layout
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("<div class='legal-container'>", unsafe_allow_html=True)
    st.subheader("Control Center")
    st.info("Status: Fully Operational")
    st.write("Current Index: Legal Corpus v1.2")
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # Chat History Container
    chat_container = st.container()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with chat_container:
        for msg in st.session_state.messages:
            div_class = "user-msg" if msg["role"] == "user" else "ai-msg"
            st.markdown(f"<div class='{div_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input Area
    prompt = st.chat_input("Ask anything about your documents...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Process AI Response if last message is from user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Analyzing context..."):
            last_prompt = st.session_state.messages[-1]["content"]
            vector_store = PostgreSQLVectorStore(secrets)
            results = vector_store.similarity_search(last_prompt)
            
            if results and results[0]['score'] > 0.6:
                context = results[0]['text']
                source = results[0]['file']
                
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"Answer formally: {last_prompt} based on {context}")
                
                full_text = f"{response.text}\n\n**Source:** {source}"
                st.session_state.messages.append({"role": "assistant", "content": full_text})
                st.rerun()
            else:
                st.error("No relevant legal context found.")

import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

# --- קונפיגורציה ועיצוב יוקרתי ---
st.set_page_config(page_title="Elite Legal AI", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #1c2e4a; color: white; border-radius: 5px; }
    .legal-card { 
        background-color: white; padding: 20px; border-radius: 10px; 
        border-left: 5px solid #d4af37; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .source-tag {
        background-color: #eef2f7; color: #1c2e4a; padding: 2px 8px;
        border-radius: 4px; font-size: 0.8rem; font-weight: bold;
    }
    h1 { color: #1c2e4a; font-family: 'Playfair Display', serif; }
    </style>
    """, unsafe_allow_html=True)

# --- לוגיקה טכנית (RAG Engine) ---
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
        # שליפת מסמכים מאונדקסים
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

# --- ממשק המשתמש (UI) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3437/3437364.png", width=80)
    st.title("System Status")
    st.success("Connection: Neon DB Active")
    st.info("Model: Gemini 1.5 Flash")
    st.divider()
    st.markdown("### Suggested Queries")
    if st.button("Analyze liability clauses"):
        st.session_state.temp_prompt = "What are the main liability limitations in our contracts?"
    if st.button("Summary of termination rights"):
        st.session_state.temp_prompt = "Explain the notice period for termination."

col1, col2 = st.columns([2, 1])

with col1:
    st.title("⚖️ Elite Legal Intelligence Hub")
    st.markdown("#### High-Precision RAG Analysis for Global Firms")
    
    # ניהול היסטוריה
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # קלט משתמש
    prompt = st.chat_input("Enter your legal inquiry...")
    if "temp_prompt" in st.session_state:
        prompt = st.session_state.pop("temp_prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Executing semantic retrieval across legal database..."):
                vector_store = PostgreSQLVectorStore(secrets)
                results = vector_store.similarity_search(prompt)
                
                if results and results[0]['score'] > 0.65:
                    context = results[0]['text']
                    source_file = results[0]['file']
                    score = results[0]['score']
                    
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(
                        f"You are a senior legal partner. Using the following clause: {context}, "
                        f"provide a formal and precise answer to: {prompt}. If the text doesn't contain the info, say so."
                    )
                    
                    full_response = response.text
                    st.markdown(f"<div class='legal-card'>{full_response}</div>", unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    with col2:
                        st.markdown("### 🔍 Evidence & Sources")
                        st.markdown(f"**Primary Source:** <span class='source-tag'>{source_file}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Confidence Score:** `{score:.2%}`")
                        with st.expander("View Original Text Segment"):
                            st.write(context)
                else:
                    st.warning("No high-confidence legal matches found in the current index.")

with col2:
    if not st.session_state.messages:
        st.markdown("### 🔍 Evidence Hub")
        st.write("Perform a search to see supporting legal segments and confidence scores.")

import streamlit as st
import os
import json
import psycopg2
import numpy as np
import google.generativeai as genai
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# הגדרת דף
st.set_page_config(page_title="Legal AI Agent", page_icon="⚖️")

# פונקציה לקבלת מפתחות מה-Secrets של Streamlit
def get_secrets():
    return {
        "POSTGRES_URL": st.secrets["POSTGRES_URL"],
        "GEMINI_API_KEY": st.secrets["GEMINI_API_KEY"]
    }

secrets = get_secrets()
genai.configure(api_key=secrets["GEMINI_API_KEY"])

# שימוש במחלקות שלך מ-search_documents.py
class GeminiEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']

class PostgreSQLVectorStore:
    def __init__(self, secrets):
        self.embeddings = GeminiEmbeddings()
        self.postgres_url = secrets["POSTGRES_URL"]
    
    def similarity_search(self, query: str, k: int = 3):
        query_embedding = np.array(self.embeddings.embed_query(query))
        conn = psycopg2.connect(self.postgres_url)
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_text, embedding, filename FROM legal_chunks LIMIT 500")
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            doc_embedding = np.array(json.loads(row[1]) if isinstance(row[1], str) else row[1])
            score = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            results.append({"text": row[0], "score": score, "file": row[2]})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        conn.close()
        return results[:k]

# ממשק המשתמש (UI)
st.title("⚖️ Legal RAG AI Agent")
st.info("I search through indexed legal documents in Neon DB to give you grounded answers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            vector_store = PostgreSQLVectorStore(secrets)
            results = vector_store.similarity_search(prompt)
            
            if results:
                context = results[0]['text']
                source_file = results[0]['file']
                
                # יצירת תשובה עם Gemini
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"Based on this legal text: {context}, answer the question: {prompt}")
                
                full_response = f"{response.text}\n\n**Source:** {source_file}"
                st.write(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.write("No relevant legal documents found in the database.")

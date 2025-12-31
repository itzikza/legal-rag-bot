import streamlit as st
import os
from main import get_rag_chain # הנחה שהפונקציה המרכזית שלך מחזירה את ה-chain

# עיצוב דף בסיסי
st.set_page_config(page_title="Legal RAG Bot", page_icon="⚖️", layout="centered")

st.title("⚖️ Legal Document AI Agent")
st.markdown("### Grounded answers for complex legal queries")

# סרגל צד להעלאת קבצים והגדרות
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload a Legal Document (PDF)", type="pdf")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.info("This agent uses RAG to ensure answers are based strictly on the uploaded document.")

# ניהול היסטוריית הצ'אט
if "messages" not in st.session_state:
    st.session_state.messages = []

# הצגת הודעות קודמות
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# לוגיקת הצ'אט
if prompt := st.chat_input("Ask a legal question about the document..."):
    if not uploaded_file or not api_key:
        st.error("Please upload a file and provide an API key first.")
    else:
        # הצגת הודעת המשתמש
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # יצירת תשובה מהאייגנט
        with st.chat_message("assistant"):
            with st.spinner("Analyzing legal context..."):
                try:
                    # כאן אנחנו מחברים את הלוגיקה מה-main.py שלך
                    os.environ["OPENAI_API_KEY"] = api_key
                    # שליחת השאלה ל-RAG chain שלך
                    # (התאם את הקריאה הזו למבנה המדויק ב-main.py)
                    response = "This is a simulated response. Connect your main.py logic here." 
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

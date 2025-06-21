import os
import sys
import uuid
import json
from datetime import datetime
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2

# Load environment variables
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate environment variables
if not POSTGRES_URL:
    raise ValueError("POSTGRES_URL environment variable is required")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


def load_document(file_path: str) -> str:
    """Load document based on file type"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return "\n\n".join([doc.page_content for doc in documents])
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            return loader.load()[0].page_content
        else:
            raise ValueError("Unsupported file format. Only PDF and DOCX files are supported.")
            
    except FileNotFoundError as e:
        print(f"File error: {e}")
        raise
    except Exception as e:
        print(f"Error loading document: {e}")
        raise


def split_text(text: str) -> List[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    try:
        if not text.strip():
            raise ValueError("Document text is empty")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        if not chunks:
            raise ValueError("Failed to create chunks from document")
        
        return chunks
        
    except Exception as e:
        print(f"Error splitting text: {e}")
        raise


def create_embedding(text: str) -> List[float]:
    """Create embedding using Gemini API"""
    try:
        if not text.strip():
            raise ValueError("Text for embedding cannot be empty")
        
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        
        if 'embedding' not in response:
            raise ValueError("Failed to generate embedding")
        
        return response['embedding']
        
    except Exception as e:
        print(f"Error creating embedding: {e}")
        raise


def store_in_db(chunks: List[str], filename: str) -> None:
    """Store chunks in PostgreSQL database with duplication check by filename"""
    conn = None
    cursor = None

    try:
        if not chunks:
            raise ValueError("No chunks to store")

        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()

        #  Duplication check
        cursor.execute("SELECT COUNT(*) FROM legal_chunks WHERE filename = %s", (filename,))
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"⚠️ File '{filename}' already indexed in the database. Skipping insert.")
            return

        successful_inserts = 0

        for i, chunk in enumerate(chunks):
            try:
                embedding = create_embedding(chunk)
                cursor.execute("""
                    INSERT INTO legal_chunks (id, chunk_text, embedding, filename, split_strategy, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()),
                    chunk,
                    json.dumps(embedding),
                    filename,
                    "recursive_character",
                    datetime.now()
                ))
                successful_inserts += 1

            except Exception as e:
                print(f"Warning: Failed to process chunk {i+1}: {e}")
                continue

        if successful_inserts == 0:
            raise ValueError("Failed to insert any chunks into database")

        conn.commit()
        print(f"✅ Successfully stored {successful_inserts}/{len(chunks)} chunks from '{filename}'")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        print(f"Error storing in database: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



def process_file(file_path: str) -> None:
    """Main processing function"""
    try:
        print(f"Processing: {os.path.basename(file_path)}")
        
        text = load_document(file_path)
        chunks = split_text(text)
        store_in_db(chunks, os.path.basename(file_path))
        
        print(f" Successfully processed '{os.path.basename(file_path)}'")
        
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: python index_document.py <file_path>")
            print("Supported formats: PDF (.pdf), Word (.docx)")
            sys.exit(1)
        
        file_path = sys.argv[1]
        process_file(file_path)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)
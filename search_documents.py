import os
import json
import psycopg2
import numpy as np
from typing import List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores.base import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate environment variables
if not POSTGRES_URL:
    raise ValueError("POSTGRES_URL environment variable is required")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)


class GeminiEmbeddings(Embeddings):
    """Gemini-based embeddings for LangChain integration."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for document chunks."""
        try:
            return [
                genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )['embedding'] for text in texts
            ]
        except Exception as e:
            print(f"Error generating document embeddings: {e}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for search query."""
        try:
            return genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_query"
            )['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []


class PostgreSQLVectorStore:
    """PostgreSQL-backed vector store with cosine similarity search."""
    
    def __init__(self, embeddings: GeminiEmbeddings):
        self.embeddings = embeddings
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        """Not implemented for read-only store."""
        raise NotImplementedError("Read-only vector store")
    
    def similarity_search(self, query: str, k: int = 1) -> List[Document]:
        """Search for k most similar documents using cosine similarity."""
        try:
            query_embedding = np.array(self.embeddings.embed_query(query))
            
            if len(query_embedding) == 0:
                print("Failed to generate query embedding")
                return []
            
            documents = self._fetch_all_documents()
            
            if not documents:
                print("No documents found in database")
                return []
            
            similarities = []
            for doc in documents:
                try:
                    similarity_score = self._cosine_similarity(query_embedding, doc["embedding"])
                    similarities.append({
                        "similarity": similarity_score,
                        "document": Document(
                            page_content=doc["text"],
                            metadata={
                                "id": doc["id"],
                                "filename": doc["filename"],
                                "strategy": doc["strategy"],
                                "similarity_score": similarity_score
                            }
                        )
                    })
                except Exception as e:
                    print(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                    continue
            
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return [item["document"] for item in similarities[:k]]
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def _fetch_all_documents(self) -> List[dict]:
        """Retrieve all document embeddings from PostgreSQL."""
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(POSTGRES_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT id, chunk_text, embedding, filename, split_strategy FROM legal_chunks LIMIT 1000")
            rows = cursor.fetchall()
            
            documents = []
            for row in rows:
                try:
                    embedding_data = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                    documents.append({
                        "id": row[0],
                        "text": row[1],
                        "embedding": np.array(embedding_data),
                        "filename": row[3],
                        "strategy": row[4]
                    })
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing embedding for document {row[0]}: {e}")
                    continue
            
            return documents
            
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error fetching documents: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            return 0.0 if norms == 0 else float(dot_product / norms)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0


def search_similar_documents(query: str, top_k: int = 5) -> List[Document]:
    """Search for similar legal documents using vector similarity."""
    if not query.strip():
        print("Query cannot be empty")
        return []
    
    try:
        embeddings = GeminiEmbeddings()
        vector_store = PostgreSQLVectorStore(embeddings)
        return vector_store.similarity_search(query, k=top_k)
    except Exception as e:
        print(f"Error in search: {e}")
        return []


if __name__ == "__main__":
    try:
        query = input("Ask me a legal question and I'll find the most relevant answers for you:\n> ")
        
        if not query.strip():
            print("Please enter a valid question.")
            exit(1)
        
        results = search_similar_documents(query)
        
        if not results:
            print("No results found. Make sure you have documents indexed in the database.")
        else:
            print(f"\nTop {len(results)} matching documents:\n")
            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                print(f"{i}. [Score: {metadata['similarity_score']:.4f}] {metadata['filename']}")
                print(f"{doc.page_content}\n")
                
    except KeyboardInterrupt:
        print("\nSearch cancelled.")
    except Exception as e:
        print(f"Application error: {e}")
        exit(1)



import chromadb
import os

def sanitize_input(text):
    """Basic input sanitization to prevent simple injection attacks."""
    return text.replace("<", "<").replace(">", ">") # Escape HTML-like characters

def initialize_chroma_client():
    """Initializes and returns a ChromaDB client with persistent storage."""
    persist_directory = os.path.join(".", "chroma_db")  # Specify the persistence directory
    client = chromadb.PersistentClient(path=persist_directory)
    return client

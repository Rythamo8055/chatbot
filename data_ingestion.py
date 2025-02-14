import io
import pdfplumber
from docx import Document
import csv
from PIL import Image
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def process_text_file(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error processing text file {file.name}: {e}")
        return None

def process_pdf_file(file):
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF file {file.name}: {e}")
        return None

def process_md_file(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error processing Markdown file {file.name}: {e}")
        return None

def chunk_text(document_content, chunk_size=500, chunk_overlap=50): # Added chunk_size parameter
    """Chunking by sentences."""
    if not document_content:
        return []
    sentences = sent_tokenize(document_content)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size: # Use chunk_size parameter
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Placeholder functions for metadata extraction
def extract_page_number(uploaded_file, chunk_index):
    return "Page number extraction not implemented yet"

def extract_section_heading(document_content, chunk_index):
    return "Section heading extraction not implemented yet"

def add_file_to_chroma(file, collection, chunk_size): # Added chunk_size parameter
    file_name = file.name
    file_type = file.type
    document_content = None

    try:
        if "text/plain" in file_type or file_name.lower().endswith(".txt"):
            document_content = process_text_file(file)
        elif "pdf" in file_type or file_name.lower().endswith(".pdf"):
            document_content = process_pdf_file(file)
        elif file_name.lower().endswith(".md"):
            document_content = process_md_file(file)
        else:
            document_content = f"Unsupported file type: {file_type} / {file_name}. Only .pdf, .txt, and .md files are processed for content."
            st.error(document_content)
            return None

        if document_content:
            chunks = chunk_text(document_content, chunk_size=chunk_size) # Pass chunk_size to chunk_text
            if chunks:
                ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
                metadatas = []
                for i in range(len(chunks)):
                    metadata = {
                        "filename": file_name,
                        "filetype": file_type,
                        "chunk_id": i,
                        "page_number": extract_page_number(file, i) if file_type == 'pdf' else None,
                        "section_heading": extract_section_heading(document_content, i) if file_type in ['md', 'txt'] else None,
                        "source": "Uploaded Files"
                    }
                    metadatas.append(metadata)

                existing_ids = collection.get(ids=ids, include=[])['ids']
                new_ids = [id for id in ids if id not in existing_ids]
                new_documents = [chunks[i] for i, id in enumerate(ids) if id in new_ids]
                new_metadatas = [metadatas[i] for i, id in enumerate(ids) if id in new_ids]

                if new_documents:
                    collection.add(
                        documents=new_documents,
                        ids=new_ids,
                        metadatas=new_metadatas
                    )
                else:
                    st.info(f"No new content to add for file '{file_name}'. Skipping.")
                return file_name
            else:
                return f"File '{file_name}' processed, but no content to add to ChromaDB."
        else:
            return None
    except Exception as e_add_chroma:
        st.error(f"Error adding file '{file_name}' to ChromaDB: {e_add_chroma}")
        return None
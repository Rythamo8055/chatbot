import streamlit as st
import chromadb
import asyncio
import google.generativeai as genai
import time
import io
import pdfplumber
from docx import Document
import csv
from PIL import Image
from data_ingestion import add_file_to_chroma, process_text_file, process_pdf_file, process_md_file, chunk_text
from query_engine import query_chroma, generate_gemini_response, expand_query
from utils import sanitize_input

# --- Page configuration ---
st.set_page_config(
    page_title="RAG Control Panel",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Materialistic and "Googlooking" Custom CSS --- (same as before)
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .st-header, .st-subheader, .st-title {
            color: #333333;
        }
        .st-file-uploader label {
            color: #1a73e8;
            font-weight: 500;
        }
        .st-button > button {
            color: white;
            background-color: #1a73e8;
            border-color: #1a73e8;
            border-radius: 4px;
        }
        .st-button > button:hover {
            background-color: #0b57d0;
            border-color: #0b57d0;
            color: white;
        }
        .st-success {
            background-color: #e6f4ea;
            color: #34a853;
            border-color: #b7e1cd;
            border-radius: 4px;
            padding: 15px;
        }
        .st-error {
            background-color: #fce8e6;
            color: #e83b28;
            border-color: #f5b3aa;
            border-radius: 4px;
            padding: 15px;
        }
        .st-file-uploader div div div:nth-child(1) {
            background-color: #f8f9fa;
            border: 1px dashed #ced4da;
            border-radius: 5px;
            padding: 20px;
        }
        .st-file-uploader div div div:nth-child(1):hover {
            background-color: #e9ecef;
            border-color: #adb5bd;
        }
        .chat-message {
            background-color: #f0f2f6;
        }
        .user-message {
            background-color: #e0e0e0;
            text-align: right;
        }
        .bot-message {
            background-color: #c9e5ff;
        }
        .st-caption {
            /* You can add styles here if needed in the future */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize ChromaDB client --- (same as before)
async def load_chroma_client():
    try:
        from utils import initialize_chroma_client
        client = chromadb.Client()
        collection_name = "gemini_rag_collection"
        try:
            collection = client.get_collection(name=collection_name)
        except chromadb.errors.InvalidCollectionException:
            collection = client.create_collection(name=collection_name)
        return collection
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        return None

# --- Google Gemini API Setup --- (same as before)
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e_gemini_init:
    st.error(f"Error initializing Gemini API: {e_gemini_init}")
    st.stop()

async def main():
    st.title("Gemini RAG Chatbot Control Panel")

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("RAG System Controls")

        # System Prompt Control
        system_prompt_input = st.text_area(
            "System Prompt",
            value="You are an expert document chatbot. Your role is to provide concise and direct answers to questions based *only* on the context provided in the retrieved documents.\n\n*   **Strictly adhere to the context.** Do not use outside information.\n*   **Be concise.** Provide direct answers without unnecessary details.\n*   If the answer is not found within the context, respond: 'Based on the provided documents, I cannot answer this question.'",
            height=250,
            help="Modify the system prompt to change Gemini's behavior.",
        )

        # Chunk Size Control
        chunk_size_input = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Adjust the size of text chunks for processing documents.",
        )

        # Temperature Control
        temperature_input = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Control Gemini's response randomness (0.0: deterministic, 1.0: highly random).",
        )
        st.markdown("---")

    collection_instance = await load_chroma_client()
    if collection_instance is None:
        st.stop()

    # File Upload Section (same as before, but now passing chunk_size_input)
    with st.expander("Upload Files to ChromaDB", expanded=False): # Set to False for better initial UI
        uploaded_files = st.file_uploader(
            "Drag and drop files here, or click to browse",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md'],
            help="Upload PDF, TXT, and MD files to add them to the knowledge base."
        )

        if uploaded_files:
            success_message_list = []
            error_message_list = []
            progress_bar = st.progress(0)
            num_files = len(uploaded_files)

            for index, uploaded_file in enumerate(uploaded_files):
                if uploaded_file is not None:
                    file_name = add_file_to_chroma(uploaded_file, collection_instance, chunk_size=chunk_size_input) # Pass chunk_size
                    if file_name:
                        if file_name.startswith("File '") and file_name.endswith("' processed, but no content to add to ChromaDB."):
                            success_message_list.append(
                                f"⚠️ File: **{uploaded_file.name}** - Uploaded, but no processable content."
                            )
                        else:
                            success_message_list.append(
                                f"✅ File: **{uploaded_file.name}** - Uploaded and added to database."
                            )
                    else:
                        error_message_list.append(
                            f"❌ File: **{uploaded_file.name}** - Error during processing."
                        )
                progress_percent = int(((index + 1) / num_files) * 100)
                progress_bar.progress(progress_percent)
                time.sleep(0.1)

            progress_bar.empty()

            if success_message_list:
                st.success("File Upload Summary:")
                for msg in success_message_list:
                    st.write(msg)
            if error_message_list:
                st.error("File Upload Issues:")
                for msg in error_message_list:
                    st.write(msg)

    # Files in Database Section (same as before)
    with st.expander("Files in Database", expanded=False):
        if collection_instance:
            try:
                ids_in_db = collection_instance.get(include=[])['ids']
                filenames = set()

                if ids_in_db:
                    for doc_id in ids_in_db:
                        parts = doc_id.split('_chunk_')
                        if len(parts) > 0:
                            filenames.add(parts[0])

                    if filenames:
                        st.write("Files currently in the database:")
                        for filename in filenames:
                            st.markdown(f"- `{filename}`")
                    else:
                        st.write("No files found in the database.")
                else:
                    st.write("No files found in the database.")

            except Exception as e_list_files:
                st.error(f"Error listing files from ChromaDB: {e_list_files}")
        else:
            st.error("ChromaDB collection not initialized.")

    # Chat Interface Section (passing system_prompt_input and temperature_input)
    st.subheader("Chat with your Documents")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Upload files and ask questions!"}]

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask questions about your documents here"):
        sanitized_prompt = sanitize_input(prompt)
        st.session_state["messages"].append({"role": "user", "content": sanitized_prompt})
        with st.chat_message("user"):
            st.markdown(sanitized_prompt)

        expanded_prompt = expand_query(sanitized_prompt)
        chroma_results = query_chroma(expanded_prompt, collection_instance)
        if chroma_results and chroma_results['documents']:
            context_docs = [doc for doc_list in chroma_results['documents'] for doc in doc_list]
            gemini_response = generate_gemini_response(sanitized_prompt, context_docs, system_prompt=system_prompt_input, temperature=temperature_input) # Pass system prompt and temperature
        else:
            gemini_response = "No relevant documents found to answer your question. Please upload files first."

        st.session_state["messages"].append({"role": "assistant", "content": gemini_response})
        with st.chat_message("assistant"):
            st.markdown(gemini_response)

    st.markdown("---")
    st.caption("Powered by Streamlit, ChromaDB, and Google Gemini - RAG Chatbot with UI Controls")

if __name__ == "__main__":
    asyncio.run(main())


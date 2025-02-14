# streamlit_app.py
import streamlit as st
import chromadb
import io
import pdfplumber
from docx import Document
import csv
from PIL import Image
import time  # For progress bar simulation
import google.generativeai as genai  # Google Gemini API

# --- Page configuration for a wider, more elegant layout ---
st.set_page_config(
    page_title="Gemini RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Materialistic and "Googlooking" Custom CSS ---
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f2f6; /* Softer background gray */
        }
        .st-header, .st-subheader, .st-title {
            color: #333333; /* Darker, more professional text for headers */
        }
        .st-file-uploader label {
            color: #1a73e8; /* Google Blue for file uploader label */
            font-weight: 500; /* Slightly bolder font-weight */
        }
        .st-button > button {
            color: white;
            background-color: #1a73e8; /* Google Blue for buttons */
            border-color: #1a73e8;
            border-radius: 4px; /* Slightly rounded buttons */
        }
        .st-button > button:hover {
            background-color: #0b57d0; /* Darker shade of Google Blue on hover */
            border-color: #0b57d0;
            color: white;
        }
        .st-success {
            background-color: #e6f4ea; /* Light green for success messages */
            color: #34a853; /* Google Green for success text */
            border-color: #b7e1cd;
            border-radius: 4px;
            padding: 15px;
        }
        .st-error {
            background-color: #fce8e6; /* Light red for error messages */
            color: #e83b28; /* Google Red for error text */
            border-color: #f5b3aa;
            border-radius: 4px;
            padding: 15px;
        }
        .st-file-uploader div div div:nth-child(1) {
            background-color: #f8f9fa; /* Lighter background for uploader area */
            border: 1px dashed #ced4da; /* Dashed border for uploader area */
            border-radius: 5px;
            padding: 20px;
        }
        .st-file-uploader div div div:nth-child(1):hover {
            background-color: #e9ecef; /* Slightly darker on hover */
            border-color: #adb5bd;
        }
        .chat-message {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #e0e0e0;
            text-align: right;
        }
        .bot-message {
            background-color: #c9e5ff;
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize ChromaDB client ---
try:
    client = chromadb.Client()
    collection_name = "gemini_rag_collection" # Changed collection name
    try:
        collection = client.get_collection(name=collection_name)
    except chromadb.errors.InvalidCollectionException:
        collection = client.create_collection(name=collection_name)
except Exception as e_chroma_init:
    st.error(f"Error initializing ChromaDB: {e_chroma_init}")
    st.stop()

# --- Google Gemini API Setup ---
#GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Ensure you have this in your secrets
genai.configure(api_key="AIzaSyBaoW_4nBrBIxNbH34UfuW0QoiwkpWvnrE")
model = genai.GenerativeModel('gemini-2.0-flash') # Or use 'gemini-pro-vision' if you want to process images

# --- File Processing Functions (same as before, but adjusted for embedding if needed) ---
def process_text_file(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        return f"Error processing text file {file.name}: {e}"

def process_pdf_file(file):
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        return f"Error processing PDF file {file.name}: {e}"

def process_docx_file(file):
    try:
        document = Document(io.BytesIO(file.read()))
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Error processing DOCX file {file.name}: {e}"

def process_csv_file(file):
    try:
        text_content = ""
        csvfile = io.StringIO(file.read().decode("utf-8"))
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            text_content += ", ".join(row) + "\n"
        return text_content
    except Exception as e:
        return f"Error processing CSV file {file.name}: {e}"

def process_image_file(file):
    try:
        img = Image.open(io.BytesIO(file.read()))
        return f"Image file: {file.name}, format: {img.format}, size: {img.size}. Image content is not directly processed into ChromaDB." # Adjusted message
    except Exception as e:
        return f"Error processing image file {file.name}: {e}"

def add_file_to_chroma(file):
    file_name = file.name
    file_type = file.type
    document_content = None

    try:
        if "text/plain" in file_type:
            document_content = process_text_file(file)
        elif "pdf" in file_type:
            document_content = process_pdf_file(file)
        elif "docx" in file_type or "officedocument.wordprocessingml.document" in file_type:
            document_content = process_docx_file(file)
        elif "csv" in file_type or "excel" in file_type:
            document_content = process_csv_file(file)
        elif "image" in file_type:
            document_content = process_image_file(file) # Still process image files to store metadata
        else:
            document_content = f"Unsupported file type: {file_type}. Only text, PDF, DOCX, CSV, and images are processed for content."

        if document_content:
            collection.add(
                documents=[document_content],
                ids=[file_name],
                metadatas=[{"filename": file_name, "filetype": file_type}]
            )
            return file_name
        else:
            return None

    except Exception as e_add_chroma:
        st.error(f"Error adding file '{file_name}' to ChromaDB: {e_add_chroma}")
        return None

def query_chroma(query_text):
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=3  # Fetch top 3 relevant documents
        )
        return results
    except Exception as e_query:
        st.error(f"Error querying ChromaDB: {e_query}")
        return None

def generate_gemini_response(query, context_documents):
    context = "\n\n".join(context_documents) # Join context documents into a single string
    prompt_parts = [
        "You are a helpful chatbot. Use the context provided to answer the question concisely and accurately.",
        "Context documents:\n" + context,
        "\nQuestion: " + query,
        "\nAnswer: "
    ]
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e_gemini:
        st.error(f"Error generating response with Gemini: {e_gemini}")
        return "Sorry, I could not generate a response at this time."


# --- Streamlit App UI ---
st.title("Gemini RAG Chatbot")

# File Upload Section
with st.expander("Upload Files to ChromaDB", expanded=True): # Using expander for better UI
    uploaded_files = st.file_uploader(
        "Drag and drop files here, or click to browse and select files",
        accept_multiple_files=True,
        type=None,  # Accept all file types
        help="You can upload text files, PDFs, DOCX, CSV, and images. Image content is not directly processed, but metadata is stored.",
    )

    if uploaded_files:
        success_message_list = []
        error_message_list = []
        progress_bar = st.progress(0)
        num_files = len(uploaded_files)

        for index, uploaded_file in enumerate(uploaded_files):
            if uploaded_file is not None:
                file_name = add_file_to_chroma(uploaded_file)
                if file_name:
                    success_message_list.append(
                        f"✅ File: **{file_name}** - Uploaded and added to ChromaDB."
                    )
                else:
                    error_message_list.append(
                        f"❌ File: **{uploaded_file.name}** -  Error during processing or unsupported type."
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

# Chat Interface Section
st.subheader("Chat with your Documents")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Upload your files and ask me questions!"}]

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the uploaded documents"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query ChromaDB and Generate Response
    chroma_results = query_chroma(prompt)
    if chroma_results and chroma_results['documents']:
        context_docs = [doc for doc_list in chroma_results['documents'] for doc in doc_list] # Flatten list of lists
        gemini_response = generate_gemini_response(prompt, context_docs)
    else:
        gemini_response = "No relevant documents found in ChromaDB to answer your question. Please upload files first."


    st.session_state["messages"].append({"role": "assistant", "content": gemini_response})
    with st.chat_message("assistant"):
        st.markdown(gemini_response)


st.markdown("---")
st.caption("Powered by Streamlit, ChromaDB, and Google Gemini - RAG Chatbot")
# streamlit_app.py
import streamlit as st
import chromadb
import io
import pdfplumber
from docx import Document
import csv
from PIL import Image
import time  # For progress bar simulation

# --- Page configuration for a wider, more elegant layout ---
st.set_page_config(
    page_title="ChromaDB File Uploader",
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
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize ChromaDB client ---
try:
    client = chromadb.Client()
    collection_name = "my_documents"
    try:
        collection = client.get_collection(name=collection_name)
    except chromadb.errors.InvalidCollectionException:
        collection = client.create_collection(name=collection_name)
except Exception as e_chroma_init:
    st.error(f"Error initializing ChromaDB: {e_chroma_init}")
    st.stop()

# --- File Processing Functions (same as before) ---
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
        return f"Image file: {file.name}, format: {img.format}, size: {img.size}. Image content is not directly processed into ChromaDB."
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
            document_content = process_image_file(file)
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

# --- Streamlit App UI ---
st.title("ChromaDB File Uploader")

with st.container():  # Main container for layout
    st.subheader("Upload your files to store them in ChromaDB.")
    st.write(
        "This application supports various file types and stores their content for further use with ChromaDB."
    )

    uploaded_files = st.file_uploader(
        "Drag and drop files here, or click to browse and select files",
        accept_multiple_files=True,
        type=None,  # Accept all file types
        help="You can upload text files, PDFs, DOCX, CSV, and images. Image content is not directly processed, but metadata is stored.",
        # placeholder="No files uploaded yet",  <- REMOVED this line
    )

    if uploaded_files:
        success_message_list = []  # Use lists to format messages better
        error_message_list = []
        progress_bar = st.progress(0)  # Initialize progress bar
        num_files = len(uploaded_files)

        for index, uploaded_file in enumerate(uploaded_files):
            if uploaded_file is not None:
                file_name = add_file_to_chroma(uploaded_file)
                if file_name:
                    success_message_list.append(
                        f"✅ File: **{file_name}** - Uploaded and added to ChromaDB."
                    )  # Success with emoji and bold name
                else:
                    error_message_list.append(
                        f"❌ File: **{uploaded_file.name}** -  Error during processing or unsupported type."
                    )  # Error with emoji and bold name
            progress_percent = int(((index + 1) / num_files) * 100)  # Calculate progress
            progress_bar.progress(progress_percent)  # Update progress bar
            time.sleep(
                0.1
            )  # Simulate processing time for visual effect (remove in real heavy processing)

        progress_bar.empty()  # Clear progress bar when done

        if success_message_list:
            st.success("File Upload Summary:")  # Success header
            for msg in success_message_list:
                st.write(msg)  # Write each success message
        if error_message_list:
            st.error("File Upload Issues:")  # Error header
            for msg in error_message_list:
                st.write(msg)  # Write each error message

    st.markdown("---")  # Visual separator
    st.caption("Powered by Streamlit and ChromaDB - Elegant File Uploader")  # More descriptive caption
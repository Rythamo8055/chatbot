import unittest
from unittest.mock import MagicMock
import streamlit as st
from data_ingestion import process_text_file, process_pdf_file, process_md_file, chunk_text, add_file_to_chroma
from query_engine import query_chroma, generate_gemini_response, expand_query, rerank_documents
from utils import sanitize_input

class TestDataIngestion(unittest.TestCase):

    def test_process_text_file(self):
        mock_file = MagicMock()
        mock_file.read.return_value = b"Test text content"
        mock_file.name = "test.txt"
        result = process_text_file(mock_file)
        self.assertEqual(result, "Test text content")

    def test_process_pdf_file(self):
        # Mocking pdfplumber is complex, so we'll just check if it runs without error
        mock_file = MagicMock()
        mock_file.read.return_value = b"PDF content"  # Dummy PDF content
        mock_file.name = "test.pdf"
        try:
            process_pdf_file(mock_file)
        except Exception as e:
            self.fail(f"process_pdf_file raised an exception: {e}")

    def test_process_md_file(self):
        mock_file = MagicMock()
        mock_file.read.return_value = b"# Test markdown content"
        mock_file.name = "test.md"
        result = process_md_file(mock_file)
        self.assertEqual(result, "# Test markdown content")

    def test_chunk_text(self):
        text = "This is a test paragraph.\n\nThis is another test paragraph."
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "This is a test paragraph.")
        self.assertEqual(chunks[1], "This is another test paragraph.")

class TestQueryEngine(unittest.TestCase):

    def test_sanitize_input(self):
        text = "<script>alert('XSS')</script>"
        sanitized_text = sanitize_input(text)
        self.assertEqual(sanitized_text, "<script>alert('XSS')</script>")

    def test_expand_query(self):
        query = "test query"
        expanded_query = expand_query(query)
        self.assertEqual(expanded_query, "test query OR test query related totest query")

    def test_rerank_documents(self):
        query = "test query"
        documents = ["short document", "longer document", "medium document"]
        reranked_documents = rerank_documents(query, documents)
        self.assertEqual(reranked_documents, ["longer document", "medium document", "short document"])

if __name__ == '__main__':
    unittest.main()

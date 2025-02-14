import streamlit as st
import chromadb
import google.generativeai as genai
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('sentence-transformers/all-mpnet-base-v2')

@st.cache_data(ttl=3600)
def query_chroma(query_text, _collection):
    try:
        results = _collection.query(
            query_texts=[query_text],
            n_results=10
        )
        documents = [doc for doc_list in results['documents'] for doc in doc_list]
        metadatas = [meta for meta_list in results['metadatas'] for meta in meta_list]

        document_pairs = [(query_text, doc) for doc in documents]
        rerank_scores = reranker.predict(document_pairs)

        ranked_docs_with_scores = sorted(zip(documents, metadatas, rerank_scores), key=lambda x: x[2], reverse=True)
        reranked_documents = [doc for doc, meta, score in ranked_docs_with_scores]
        reranked_metadatas = [meta for doc, meta, score in ranked_docs_with_scores]

        top_3_documents = reranked_documents[:3]
        top_3_metadatas = reranked_metadatas[:3]

        results['documents'] = [top_3_documents]
        results['metadatas'] = [top_3_metadatas]

        return results
    except Exception as e_query:
        st.error(f"Error querying ChromaDB: {e_query}")
        return None

def generate_gemini_response(query, context_documents, system_prompt, temperature): # Added system_prompt and temperature
    context = "\\n\\n".join(context_documents)
    prompt_parts = [
        system_prompt, # Now using dynamic system prompt
        "Context documents:\\n" + context,
        "\\nQuestion: " + query,
        "\\nAnswer: "
    ]
    try:
        response = genai.GenerativeModel('gemini-2.0-flash').generate_content(
            prompt_parts,
            generation_config=genai.types.GenerationConfig(temperature=temperature) # Pass temperature here
        )
        return response.text
    except Exception as e_gemini:
        st.error(f"Error generating response with Gemini: {e_gemini}")
        return "Sorry, I could not generate a response at this time."

def expand_query(query):
    expanded_query = query + " OR " + query + " related to" + query
    return expanded_query
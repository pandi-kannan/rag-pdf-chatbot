"""
app.py
------
Streamlit UI for the RAG chatbot. Each uploaded PDF gets its own
isolated collection, so different documents never mix together.

Run:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src.ingest import load_pdf, split_into_chunks, build_vector_store, sanitize_collection_name, PERSIST_DIR
from src.query import load_vector_store, retrieve_context, generate_answer, list_collections

load_dotenv()

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄")
st.title("📄 RAG PDF Chatbot")
st.caption("Upload a PDF, then ask questions grounded in its content. Each document is kept separate.")

if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

# --- Sidebar: upload + ingest ---
with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        pdf_path = os.path.join("data", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        collection_name = sanitize_collection_name(uploaded_file.name)

        if st.button("Ingest this PDF"):
            with st.spinner("Reading, chunking, and embedding the PDF..."):
                pages = load_pdf(pdf_path)
                chunks = split_into_chunks(pages)
                build_vector_store(chunks, collection_name=collection_name)
            st.session_state.active_collection = collection_name
            st.success(f"Ingested {len(chunks)} chunks for '{uploaded_file.name}'.")

    st.divider()
    st.header("2. Or pick a previously ingested document")
    if os.path.exists(PERSIST_DIR):
        existing = list_collections()
        if existing:
            chosen = st.selectbox("Existing documents", ["(none)"] + existing)
            if chosen != "(none)":
                st.session_state.active_collection = chosen

# --- Main: ask questions ---
st.header("3. Ask a question")

if not st.session_state.active_collection:
    st.info("Upload and ingest a PDF, or pick an existing document, from the sidebar.")
else:
    st.caption(f"Currently querying: **{st.session_state.active_collection}**")
    question = st.text_input("Your question")

    if question:
        with st.spinner("Retrieving relevant context and generating an answer..."):
            vectordb = load_vector_store(st.session_state.active_collection)
            context, sources = retrieve_context(vectordb, question)
            answer = generate_answer(question, context)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Show retrieved source chunks"):
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}] Page {page}**")
                st.write(doc.page_content)
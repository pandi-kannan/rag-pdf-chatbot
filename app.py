"""
app.py
------
Simple Streamlit UI for the RAG chatbot.
This is optional -- the CLI (src/query.py) works fine on its own,
but a UI makes the project much easier to demo in interviews.

Run:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src.ingest import load_pdf, split_into_chunks, build_vector_store, PERSIST_DIR
from src.query import load_vector_store, retrieve_context, generate_answer

load_dotenv()

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄")
st.title("📄 RAG PDF Chatbot")
st.caption("Upload a PDF, then ask questions grounded in its content.")

# --- Sidebar: upload + ingest ---
with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        pdf_path = os.path.join("data", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Ingest this PDF"):
            with st.spinner("Reading, chunking, and embedding the PDF..."):
                pages = load_pdf(pdf_path)
                chunks = split_into_chunks(pages)
                build_vector_store(chunks)
            st.success(f"Ingested {len(chunks)} chunks. You can now ask questions.")

# --- Main: ask questions ---
st.header("2. Ask a question")

if not os.path.exists(PERSIST_DIR):
    st.info("Upload and ingest a PDF first (left sidebar).")
else:
    question = st.text_input("Your question")

    if question:
        with st.spinner("Retrieving relevant context and generating an answer..."):
            vectordb = load_vector_store()
            context, sources = retrieve_context(vectordb, question)
            answer = generate_answer(question, context)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Show retrieved source chunks"):
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}] Page {page}**")
                st.write(doc.page_content)
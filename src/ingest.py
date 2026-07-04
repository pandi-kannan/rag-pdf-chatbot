"""
ingest.py
----------
Step 1 of the RAG pipeline: load a PDF, split it into chunks,
embed each chunk, and store the embeddings in a local Chroma vector database.

Run this once whenever you add/change the source PDF.

Usage:
    python src/ingest.py --pdf data/mydoc.pdf
"""

import argparse
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free, runs locally


def load_pdf(pdf_path: str):
    """Load a PDF and return a list of LangChain Document objects (one per page)."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {pdf_path}")
    return pages


def split_into_chunks(pages, chunk_size: int = 800, chunk_overlap: int = 120):
    """
    Split pages into smaller overlapping chunks.
    chunk_overlap keeps some context between chunks so we don't lose meaning
    at the boundary of a split.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks, persist_dir: str = PERSIST_DIR):
    """Embed each chunk and store it in a persistent Chroma vector database."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Vector store built and saved to ./{persist_dir}")
    return vectordb


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG vector store.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to ingest")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"No such file: {args.pdf}")

    pages = load_pdf(args.pdf)
    chunks = split_into_chunks(pages)
    build_vector_store(chunks)

    print("\nIngestion complete. You can now run: python src/query.py")


if __name__ == "__main__":
    main()
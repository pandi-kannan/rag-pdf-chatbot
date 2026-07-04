"""
query.py
--------
Step 2 of the RAG pipeline: given a user question, retrieve the most
relevant chunks from the vector store, stuff them into a prompt, and
call the LLM to generate a grounded answer.

Run this after ingest.py has been run at least once.

Usage:
    python src/query.py
    (then type questions interactively)
"""

import os
import chromadb
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # reads ANTHROPIC_API_KEY from .env

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4  # how many chunks to retrieve per question

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.
If the answer is not contained in the context, say "I don't have enough information to answer that."
Do not make up information that isn't in the context.

Context:
{context}

Question: {question}

Answer:"""


def load_vector_store(collection_name: str, persist_dir: str = PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return vectordb


def list_collections(persist_dir: str = PERSIST_DIR):
    """List all document collections currently stored."""
    client = chromadb.PersistentClient(path=persist_dir)
    return [c.name for c in client.list_collections()]


def retrieve_context(vectordb, question: str, k: int = TOP_K):
    """Find the k most relevant chunks for the question."""
    results = vectordb.similarity_search(question, k=k)
    context = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context, results


def generate_answer(question: str, context: str):
    """Call Claude with the retrieved context to generate a grounded answer."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content


def answer_question(vectordb, question: str, show_sources: bool = True):
    context, sources = retrieve_context(vectordb, question)
    answer = generate_answer(question, context)

    print(f"\nAnswer: {answer}\n")

    if show_sources:
        print("Sources used (top chunks retrieved):")
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i}] page {page}: {preview}...")

    return answer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query a RAG collection.")
    parser.add_argument("--collection", required=False, help="Which document's collection to query")
    args = parser.parse_args()

    if not os.path.exists(PERSIST_DIR):
        print("No vector store found. Run ingest.py first:")
        print("  python src/ingest.py --pdf data/yourfile.pdf")
        return

    collection_name = args.collection
    if not collection_name:
        available = list_collections()
        if not available:
            print("No collections found. Run ingest.py first.")
            return
        if len(available) == 1:
            collection_name = available[0]
            print(f"Using collection: '{collection_name}'")
        else:
            print("Multiple documents found. Choose one:")
            for i, name in enumerate(available, 1):
                print(f"  {i}. {name}")
            choice = input("Enter number: ").strip()
            collection_name = available[int(choice) - 1]

    vectordb = load_vector_store(collection_name)
    print("RAG chatbot ready. Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer_question(vectordb, question)


if __name__ == "__main__":
    main()
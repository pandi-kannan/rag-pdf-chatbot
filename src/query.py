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


def load_vector_store(persist_dir: str = PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectordb


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
    if not os.path.exists(PERSIST_DIR):
        print("No vector store found. Run ingest.py first:")
        print("  python src/ingest.py --pdf data/yourfile.pdf")
        return
    vectordb = load_vector_store()
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
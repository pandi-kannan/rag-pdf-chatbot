# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in
the content of an uploaded PDF. Built to understand and demonstrate the core RAG
pipeline: document ingestion, chunking, embedding, vector search, and LLM-based
answer generation.

## How it works

1. **Ingest** — A PDF is loaded and split into overlapping text chunks.
2. **Embed** — Each chunk is converted into a vector using a local sentence-transformer
   embedding model (no external embedding API needed).
3. **Store** — Vectors are stored in a local Chroma vector database.
4. **Retrieve** — When a user asks a question, the question is embedded and the most
   semantically similar chunks are retrieved from the vector store.
5. **Generate** — The retrieved chunks are passed to Claude (Anthropic API) as context,
   producing an answer grounded in the source document instead of relying on the
   model's general knowledge alone.

## Tech stack

- **Python**
- **LangChain** — orchestration of the ingestion and retrieval pipeline
- **ChromaDB** — local vector database
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, free and offline
- **Anthropic API (Claude)** — answer generation
- **Streamlit** — optional web UI for demoing the chatbot

## Project structure

rag-chatbot/
├── data/              # place source PDFs here
├── src/
│   ├── ingest.py      # loads, chunks, and embeds a PDF into the vector store
│   └── query.py        # retrieves context and generates answers (CLI)
├── app.py             # Streamlit web UI
├── requirements.txt
└── .env.example

## Setup

```bash
# 1. Clone and enter the project
cd rag-chatbot

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Anthropic API key
cp .env.example .env
# then edit .env and paste your key
```

## Usage

**Option A — Command line**

```bash
# Ingest a PDF (run once per document)
python src/ingest.py --pdf data/yourfile.pdf

# Ask questions interactively
python src/query.py
```

**Option B — Web UI**

```bash
streamlit run app.py
```

Upload a PDF in the sidebar, click "Ingest this PDF," then ask questions in the main panel.

## Possible extensions

- Support multiple PDFs / a full document library
- Add conversation memory for multi-turn follow-up questions
- Swap Chroma for a hosted vector DB (Pinecone, Weaviate) for production use
- Add citation highlighting so answers link back to exact source pages

## Why this project

Built to understand Retrieval-Augmented Generation (RAG) end-to-end — the core
pattern behind most production LLM applications that need to answer questions
using private or up-to-date data the base model wasn't trained on.
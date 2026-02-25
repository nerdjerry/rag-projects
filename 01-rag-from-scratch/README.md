# 01 — RAG From Scratch 🔍📄

Build a **Retrieval-Augmented Generation** pipeline from the ground up using LangChain, FAISS, and a local or cloud LLM.

---

## What Is RAG and Why Does It Matter?

Large Language Models (LLMs) like GPT-4 or Llama are incredibly capable, but they have two big limitations:

1. **Knowledge cutoff** — They only know what was in their training data.  Ask about your company's internal docs and they'll guess (or "hallucinate").
2. **No source attribution** — A plain LLM can't tell you *where* it got its answer.

**RAG solves both problems.**  Instead of asking the LLM to answer from memory, we first *retrieve* the most relevant pieces of your own documents and hand them to the LLM as context.  The model then generates an answer grounded in that context — and we can point back to the exact source.

In plain English:

> **RAG = "Look it up, then answer."**

---

## High-Level Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Your Docs   │────▶│   Chunker    │────▶│   Embedding      │
│ (PDF/TXT/    │     │ (split into  │     │   Model          │
│  DOCX)       │     │  small parts)│     │ (text → vectors) │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │   FAISS Vector   │
                                          │   Store (index)  │
                                          └────────┬─────────┘
                                                   │
                        User Question ─────────────┤
                                                   ▼
                                          ┌──────────────────┐
                                          │   Retriever      │
                                          │ (top-k similar   │
                                          │  chunks)         │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │   LLM Generator  │
                                          │ (answer using    │
                                          │  context only)   │
                                          └──────────────────┘
```

---

## Tech Stack

| Component           | Tool / Library                        | Why                                  |
|---------------------|---------------------------------------|--------------------------------------|
| Document loading    | LangChain Loaders                     | Unified API for PDF, TXT, DOCX       |
| Text splitting      | RecursiveCharacterTextSplitter        | Smart chunking at natural boundaries  |
| Embeddings          | all-MiniLM-L6-v2 (Sentence Transformers) | Free, local, fast                 |
| Vector store        | FAISS                                 | Fast similarity search, no server     |
| LLM (cloud)        | OpenAI gpt-3.5-turbo                  | High quality, easy to set up          |
| LLM (local)        | Ollama (Llama 3, etc.)                | Free, private, no API key needed      |
| Orchestration       | LangChain RetrievalQA                 | Ties retrieval + generation together  |

---

## Folder Structure

```
01-rag-from-scratch/
├── main.py                  # Entry point — runs the full pipeline
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
├── README.md                # You are here
├── data/
│   └── sample_docs/         # Drop your .pdf, .txt, .docx files here
├── src/
│   ├── __init__.py
│   ├── document_loader.py   # Step 1: Load files into Document objects
│   ├── chunker.py           # Step 2: Split documents into chunks
│   ├── embedder.py          # Step 3: Convert text to vectors
│   ├── vector_store.py      # Step 4: Store & search vectors with FAISS
│   ├── retriever.py         # Step 5: Find relevant chunks for a query
│   └── generator.py         # Step 6: Generate answers with an LLM
└── faiss_index/             # Auto-created — saved FAISS index
```

---

## Setup Instructions

### 1. Create a virtual environment

```bash
cd 01-rag-from-scratch
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and choose your LLM backend:

- **OpenAI** — paste your API key into `OPENAI_API_KEY`.
- **Ollama (free, local)** — set `USE_OLLAMA=true` and make sure Ollama is running (`ollama serve`) with a pulled model (`ollama pull llama3`).

### 4. Add your documents

Drop `.pdf`, `.txt`, or `.docx` files into the `data/` folder (or `data/sample_docs/`).

---

## Running the Pipeline

```bash
python main.py
```

The script will:

1. Load every document in `data/`.
2. Split them into overlapping chunks.
3. Convert chunks to embeddings (first run downloads the model).
4. Build a FAISS index and save it to `faiss_index/`.
5. Prompt you for a question — or use a default sample.
6. Retrieve the most relevant chunks and generate an answer.

On subsequent runs the saved FAISS index is reloaded automatically.  
Delete `faiss_index/` to force a rebuild (e.g., after adding new documents).

---

## How to Add Your Own Documents

1. Place files in `data/` or `data/sample_docs/`.
2. Supported formats: `.pdf`, `.txt`, `.docx`.
3. Delete the `faiss_index/` folder so the index is rebuilt with the new content.
4. Run `python main.py`.

---

## Sample Questions to Try

```
What are the main topics covered in these documents?
Summarize the key points from the uploaded files.
What does the document say about <specific topic>?
```

---

## How to Verify the LLM Uses Your Documents

1. **Check the retrieved chunks** — the pipeline prints the top-k chunks before generating an answer.  Compare them to your source files.
2. **Enable debug mode** — answers include source document metadata so you can trace every claim back to a file and page.
3. **Ask about something NOT in your documents** — the LLM should respond with *"I don't have enough information to answer that"* instead of making something up.  This confirms the system prompt is working.

---

## Configuration

All tuneable parameters live in `.env`:

| Variable        | Default   | Description                                  |
|-----------------|-----------|----------------------------------------------|
| `CHUNK_SIZE`    | 500       | Max characters per chunk                      |
| `CHUNK_OVERLAP` | 50        | Overlapping characters between chunks         |
| `TOP_K`         | 3         | Number of chunks retrieved per question       |
| `USE_OLLAMA`    | false     | Set to `true` for local Ollama models         |
| `OLLAMA_MODEL`  | llama3    | Which Ollama model to use                     |

---

## License

This project is part of the **GenAI RAG Projects** series and is intended for educational use.

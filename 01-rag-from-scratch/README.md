# RAG from Scratch 🔍

A beginner-friendly implementation of Retrieval-Augmented Generation (RAG) built step-by-step using LangChain, FAISS, and HuggingFace embeddings. Every file is heavily commented to explain *why* each piece exists, not just *what* it does.

---

## What is RAG and Why Does It Matter?

**The problem with plain LLMs:** Large Language Models like GPT-4 are trained on data up to a certain cutoff date, and they have no knowledge of *your* private documents — your company's policy manuals, your research papers, your product documentation. If you ask GPT-4 "What is the refund policy in our internal handbook?", it simply doesn't know.

**What RAG does:** RAG (Retrieval-Augmented Generation) solves this by giving the LLM access to your documents *at query time*. Instead of retraining the model (expensive, slow), you store your documents in a searchable vector database. When a user asks a question, you retrieve the most relevant passages and include them in the LLM's prompt. The LLM reads those passages and answers *based on your documents*.

**Why it matters:** RAG is currently the dominant architecture for production AI Q&A systems. It's cost-effective (no retraining), updatable (just add documents to the database), and auditable (you can see exactly which document chunks informed each answer). Understanding RAG from scratch gives you the foundation to build everything from customer support bots to internal knowledge assistants.

---

## Architecture

```
YOUR DOCUMENTS (PDF / TXT / DOCX)
         │
         ▼
  ┌─────────────┐
  │  1. LOAD    │  Read files from disk into LangChain Document objects
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  2. CHUNK   │  Split large docs into ~500-char overlapping pieces
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  3. EMBED   │  Convert each chunk → 384-dim vector (all-MiniLM-L6-v2)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  4. INDEX   │  Store vectors in FAISS (saved to disk for reuse)
  └──────┬──────┘
         │
         │                    USER QUESTION
         │                         │
         │                         ▼
         │                  ┌─────────────┐
         │                  │  5. EMBED   │  Embed question → vector
         │                  └──────┬──────┘
         │                         │
         ▼                         ▼
  ┌─────────────────────────────────────┐
  │         FAISS SIMILARITY SEARCH     │  Find top-k most similar chunks
  └──────────────────┬──────────────────┘
                     │
                     ▼
             TOP-k RELEVANT CHUNKS
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │     6. GENERATE (LLM + Prompt)      │  LLM reads chunks + question
  └──────────────────┬──────────────────┘
                     │
                     ▼
              GROUNDED ANSWER ✅
```

---

## Tech Stack

| Component         | Library / Tool                          | Purpose                                  |
|-------------------|-----------------------------------------|------------------------------------------|
| Document loading  | `langchain-community` loaders           | Read PDF, TXT, DOCX files                |
| Text splitting    | `langchain` RecursiveCharacterTextSplitter | Split docs into overlapping chunks    |
| Embeddings        | `sentence-transformers` (HuggingFace)   | Convert text → vectors (free, local)     |
| Vector database   | `faiss-cpu`                             | Fast similarity search over embeddings   |
| LLM               | OpenAI GPT-3.5/4 or local Ollama        | Generate answers from retrieved context  |
| Orchestration     | `langchain` RetrievalQA chain           | Tie retrieval + generation together      |
| Env management    | `python-dotenv`                         | Load API keys from `.env` file           |

---

## Step-by-Step Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ First install may take a few minutes. `faiss-cpu` and `sentence-transformers` are the largest packages.

### 3. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and replace `your_openai_api_key_here` with your actual key from [platform.openai.com](https://platform.openai.com/api-keys).

```
OPENAI_API_KEY=sk-...your-key-here...
```

> 💡 **No OpenAI account?** Use a local model with Ollama — see [Using Ollama](#using-ollama-no-api-key-needed) below.

### 4. Add your documents

Drop any `.pdf`, `.txt`, or `.docx` files into:

```
data/sample_docs/
```

The more documents you add, the more the system can answer. Start with a few text files to test.

### 5. Run it!

```bash
# Interactive mode — asks questions in a loop
python main.py

# Single question mode
python main.py --question "What are the main topics in these documents?"

# Debug mode — shows retrieved chunks and full LLM prompt
python main.py --debug --question "What is the refund policy?"
```

---

## How to Add Your Own Documents

Just drop files into `data/sample_docs/`. The loader automatically detects file types:

| File type | Support | Notes |
|-----------|---------|-------|
| `.pdf`    | ✅      | Each page becomes a separate Document |
| `.txt`    | ✅      | Entire file is one Document |
| `.docx`   | ✅      | Entire file is one Document |
| `.csv`    | ❌      | Not supported (yet) |

**After adding new documents**, delete the cached FAISS index so it gets rebuilt:

```bash
rm -rf faiss_index/
python main.py
```

---

## How to Verify the LLM Uses Your Documents

This is the most important test for any RAG system — make sure it's actually reading *your* documents and not falling back on general knowledge.

**Step 1:** Put a document with a very specific, obscure fact in `data/sample_docs/`. For example, create `test.txt` containing:

```
The Zorbax Protocol was established in 2019 by Dr. Eleanor Voss.
The protocol requires three phases: initialization, calibration, and review.
```

**Step 2:** Ask the system about it:
```bash
python main.py --question "Who established the Zorbax Protocol?"
```

**Expected good result:**
```
Answer: Dr. Eleanor Voss established the Zorbax Protocol in 2019.
Sources: data/sample_docs/test.txt
```

**Step 3:** Ask about something NOT in any document:
```bash
python main.py --question "What is the capital of Australia?"
```

**Expected good result:**
```
Answer: I don't know based on the provided documents.
```

If the second answer returns "Canberra" (from general knowledge), the system is hallucinating — check that your prompt template in `src/generator.py` is being applied correctly.

---

## Using Ollama (No API Key Needed)

[Ollama](https://ollama.com) lets you run LLMs locally for free.

```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull a model
ollama pull llama3      # ~4GB download
ollama pull mistral     # ~4GB download, often faster

# 3. Run with Ollama
python main.py --model ollama/llama3
python main.py --model ollama/mistral --question "Summarize the documents"
```

---

## Beginner Tips

### What happens if chunk_size is too large or too small?

| Setting | Effect |
|---------|--------|
| **chunk_size too large** (e.g., 2000) | Fewer chunks, less precise retrieval. The LLM receives a lot of text, most of which may be irrelevant to the question. |
| **chunk_size too small** (e.g., 50)  | Thousands of tiny chunks. Each chunk lacks context — a sentence like "See the above section" becomes meaningless on its own. |
| **Sweet spot** (300–800 chars)        | Roughly 1–2 paragraphs. Enough context to be meaningful, small enough to be precise. |

### Why cosine similarity beats keyword search

Traditional search (e.g., `grep`, SQL `LIKE`) requires exact word matches. Search for "car" and you won't find documents that say "automobile" or "vehicle".

Semantic search (cosine similarity over embeddings) understands *meaning*:
- "car", "automobile", "vehicle", "sedan" → all have very similar embeddings
- You can ask "What's the fastest way to travel?" and find chunks about "high-speed rail" or "airplane travel" — no exact keyword overlap needed

### What does k mean in top-k retrieval?

`k` is the number of document chunks retrieved per question.

- **k=1**: Only the single best match. Very precise but may miss relevant context.
- **k=3** (default): A good balance. Captures the primary answer + nearby supporting text.
- **k=10**: Comprehensive but may include loosely related chunks that dilute the LLM's focus.

Use `--k 5` on the command line to experiment. If the LLM keeps saying "I don't know" on questions you know are in the docs, try increasing k.

---

## Troubleshooting

### `OPENAI_API_KEY is not set`
```bash
cp .env.example .env
# edit .env and add your key
```

### `No documents were loaded`
Make sure you have files in `data/sample_docs/`. Only `.pdf`, `.txt`, and `.docx` are supported.

### `FileNotFoundError: data/sample_docs does not exist`
```bash
mkdir -p data/sample_docs
# then add your files
```

### `Error: Connection refused` (Ollama)
Make sure Ollama is running:
```bash
ollama serve
```

### `Model not found` (Ollama)
Pull the model first:
```bash
ollama pull llama3
```

### Answers seem wrong or generic
1. Run with `--debug` to see which chunks are being retrieved
2. Check the sources printed after each answer — are they the right files?
3. Try deleting `faiss_index/` and rebuilding — you may have stale embeddings
4. Try increasing `--k` to retrieve more context

### `pip install` fails on `faiss-cpu`
On some systems you may need to install build tools:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```

---

## Project Structure

```
01-rag-from-scratch/
├── README.md                 ← You are here
├── requirements.txt          ← Python dependencies
├── .env.example              ← Template for your API keys
├── main.py                   ← Entry point — ties all 6 steps together
├── data/
│   └── sample_docs/          ← Drop your .pdf/.txt/.docx files here
└── src/
    ├── __init__.py           ← Makes src/ a Python package
    ├── document_loader.py    ← Step 1: Load documents from disk
    ├── chunker.py            ← Step 2: Split documents into chunks
    ├── embedder.py           ← Step 3: Convert text to vectors
    ├── vector_store.py       ← Step 4: Store/search vectors with FAISS
    ├── retriever.py          ← Step 5: Retrieve relevant chunks
    └── generator.py          ← Step 6: Generate answers with LLM
```

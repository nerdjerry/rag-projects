# Project 4 — Multimodal RAG

> **Go beyond plain text.** This project builds a Retrieval-Augmented
> Generation system that understands **text**, **images**, and **tables**
> inside PDF documents.

---

## Architecture

```
                         ┌──────────────┐
                         │   PDF File   │
                         └──────┬───────┘
                                │
                       ┌────────▼────────┐
                       │ Multimodal      │
                       │ Parser          │
                       └──┬─────┬─────┬──┘
                          │     │     │
               ┌──────────┘     │     └──────────┐
               ▼                ▼                 ▼
        ┌────────────┐  ┌────────────┐  ┌──────────────┐
        │ Text       │  │ Images     │  │ Tables       │
        │ Chunks     │  │ (PNG)      │  │ (CSV)        │
        └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
              │               │                 │
              │         ┌─────▼──────┐   ┌──────▼───────┐
              │         │ Vision LLM │   │ LLM Describe │
              │         │ → Caption  │   │ → Summary    │
              │         └─────┬──────┘   └──────┬───────┘
              │               │                 │
        ┌─────▼──────┐ ┌─────▼──────┐  ┌───────▼──────┐
        │ Text       │ │ Image      │  │ Table        │
        │ FAISS      │ │ FAISS      │  │ FAISS        │
        │ Index      │ │ Index      │  │ Index        │
        └─────┬──────┘ └─────┬──────┘  └───────┬──────┘
              │               │                 │
              └───────┐       │       ┌─────────┘
                      ▼       ▼       ▼
                ┌─────────────────────────┐
                │     Query Router        │
                │  "Which indexes to      │
                │   search?"              │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │   Multi-Retriever       │
                │   Merge + Rank          │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │   Generator (LLM)       │
                │   → Final Answer        │
                └─────────────────────────┘
```

---

## What Is "Multimodal" RAG?

Traditional RAG only handles text. But real-world documents contain much
more:

| Modality | Examples | What text-only RAG misses |
|----------|---------|--------------------------|
| **Text** | Paragraphs, headings, lists | *(handled)* |
| **Images** | Charts, diagrams, photos, screenshots | A bar chart showing revenue trends is invisible to text search |
| **Tables** | Financial data, comparison matrices, specs | Row/column structure is lost when flattened to plain text |
| **Cross-modal** | "Compare the chart on page 3 with Table 2" | Requires reasoning across modalities |

### Examples of questions this project can answer

- *"What trend does the line chart on page 5 show?"* — needs image
  understanding
- *"What was Q3 revenue?"* — needs table search
- *"Summarize the introduction and relate it to Figure 1"* — needs text +
  image

---

## Model Comparison

| Capability | OpenAI (GPT-4o) | Ollama (local) |
|------------|-----------------|----------------|
| Text generation | ✅ Excellent | ✅ Good (llama3) |
| Image captioning | ✅ GPT-4o vision | ✅ LLaVA model |
| Table description | ✅ Excellent | ✅ Good |
| Cost | 💰 Pay-per-token | 🆓 Free |
| Privacy | ☁️ Cloud | 🔒 Local |
| Speed | ⚡ Fast | 🐢 Depends on hardware |

---

## Cost Considerations

| Operation | Approximate Cost (GPT-4o) |
|-----------|--------------------------|
| Parse PDF | Free (local library) |
| Caption 1 image | ~$0.01–0.03 |
| Describe 1 table | ~$0.001–0.005 |
| Embed chunks | Free (local model) |
| 1 query + answer | ~$0.01–0.03 |

**Cost-saving tips:**
- Use Ollama for development (free, runs locally)
- Cache image captions and table descriptions (they don't change)
- Use the query router to avoid searching unnecessary indexes
- Use `all-MiniLM-L6-v2` for embeddings (free, fast, local)

---

## Setup

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.ai) for free local models

### Installation

```bash
# 1. Navigate to the project directory
cd 04-multimodal-rag

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key (or set USE_OLLAMA=true)
```

### If Using Ollama (Free, Local)

```bash
# Install Ollama from https://ollama.ai, then:
ollama pull llama3        # text model
ollama pull llava         # vision model (for image captioning)

# In your .env:
USE_OLLAMA=true
OLLAMA_MODEL=llama3
VISION_MODEL=llava
```

---

## Usage

### 1. Ingest a PDF Document

```bash
python main.py --pdf data/sample_docs/your_document.pdf
```

This will:
- Extract text, images, and tables from the PDF
- Caption each image with a vision model
- Describe each table with an LLM
- Build three FAISS indexes (text, image, table)

### 2. Ask Questions (Interactive Mode)

```bash
python main.py
```

```
You: What does the chart on page 3 show?
Assistant: The chart on page 3 is a bar chart showing quarterly revenue...

You: What was the total revenue in Q3?
Assistant: According to the financial table, Q3 revenue was $142M...

You: quit
```

### 3. Single Query Mode

```bash
python main.py --query "Summarize the key findings"
```

---

## How to Add Your Own Documents

1. Place your PDF file(s) in `data/sample_docs/`
2. Run `python main.py --pdf data/sample_docs/your_file.pdf`
3. Start asking questions with `python main.py`

**Supported formats:** PDF files with any combination of text, images, and
tables.

**Tips for best results:**
- Scanned PDFs (image-only) need OCR — consider adding `pytesseract`
- High-resolution PDFs yield better image extraction
- Tables with clear grid lines are extracted more reliably

---

## Project Structure

```
04-multimodal-rag/
├── main.py                    # Entry point — ingest & query
├── requirements.txt           # Python dependencies
├── .env.example               # Configuration template
├── README.md                  # This file
├── data/
│   ├── sample_docs/           # Put your PDFs here
│   ├── extracted/
│   │   ├── images/            # Extracted images (PNG)
│   │   └── tables/            # Extracted tables (CSV)
│   └── indexes/               # FAISS indexes (auto-created)
│       ├── text_index/
│       ├── image_index/
│       └── table_index/
└── src/
    ├── __init__.py
    ├── multimodal_parser.py   # PDF → text + images + tables
    ├── text_indexer.py        # Chunk & embed text → FAISS
    ├── image_processor.py     # Image → caption via vision LLM
    ├── image_indexer.py       # Caption embeddings → FAISS
    ├── table_processor.py     # Table → NL description via LLM
    ├── table_indexer.py       # Description embeddings → FAISS
    ├── query_router.py        # Classify query → route to indexes
    ├── multi_retriever.py     # Search + merge + rank results
    └── generator.py           # Generate final answer from context
```

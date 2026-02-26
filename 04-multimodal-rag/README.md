# 04 — Multimodal RAG

A Retrieval-Augmented Generation system that understands **text, images, and tables** inside PDF documents.

> A text-only RAG can't answer *"What does the architecture diagram show?"* — but this system can.
> It extracts every content type from the document, builds a dedicated search index per modality,
> routes each query to the right index, and generates an answer that explicitly cites whether the
> information came from a paragraph, a chart, or a data table.

---

## What "Multimodal" Means

| Query | Text-only RAG | This System |
|---|---|---|
| "Explain the authentication flow" | ✅ Can answer | ✅ Can answer |
| "What does the flowchart in section 3 show?" | ❌ Cannot answer | ✅ Captions image, answers from description |
| "What was Q4 revenue?" | ⚠️ Only if table was also in prose | ✅ Extracts table, generates description |
| "Summarise all key findings" | ✅ Partial | ✅ Draws from text + images + tables |

---

## Architecture

```
PDF Document
     │
     ▼
┌────────────────────┐
│  multimodal_parser │  ── pdfplumber extracts text / tables / images
└────────┬───────────┘
         │
    ┌────┴─────────────────────────────────┐
    ▼                ▼                     ▼
┌─────────┐   ┌────────────┐    ┌─────────────────┐
│  Text   │   │   Images   │    │     Tables      │
│ Blocks  │   │  (PNG files)│    │ (list of lists) │
└────┬────┘   └─────┬──────┘    └────────┬────────┘
     │              │                    │
     │         GPT-4V caption       LLM description
     │              │                    │
     ▼              ▼                    ▼
┌──────────┐ ┌────────────┐   ┌──────────────────┐
│  FAISS   │ │   FAISS    │   │     FAISS        │
│ Text Idx │ │ Image Idx  │   │   Table Idx      │
└────┬─────┘ └─────┬──────┘   └────────┬─────────┘
     └─────────────┴─────────────────── ┘
                         │
                  ┌──────┴──────┐
                  │ Query Router│  ── classifies query → TEXT / IMAGE / TABLE / ALL
                  └──────┬──────┘
                         │
                  ┌──────┴──────┐
                  │Multi-Retriev│  ── fetches top-k from relevant indexes, merges
                  └──────┬──────┘
                         │
                  ┌──────┴──────┐
                  │  Generator  │  ── GPT-4 builds final answer from mixed context
                  └─────────────┘
```

---

## Model Comparison

| Modality | Model | Why |
|---|---|---|
| Text | all-MiniLM-L6-v2 | Fast, free, runs locally, strong retrieval quality |
| Images | GPT-4V (`gpt-4-vision-preview`) | Understands visual content — charts, diagrams, photos |
| Tables | GPT-3.5 / GPT-4 | Strong at structured data reasoning; converts rows to prose |
| Generation | GPT-4 | Best reasoning across mixed text / image / table context |

**Alternative (cost-free images):** [LLaVA](https://ollama.com/library/llava) via Ollama runs locally and produces comparable captions without API charges.

---

## Setup

### 1. Clone and enter the project
```bash
cd 04-multimodal-rag
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### 5. Add a PDF document
```bash
cp /path/to/your/document.pdf data/sample_docs/
```

---

## Usage

### Ask a single question
```bash
python main.py --file data/sample_docs/annual_report.pdf \
               --query "What was Q4 revenue?"
```

### Skip image captioning during development (saves GPT-4V cost)
```bash
python main.py --file data/sample_docs/annual_report.pdf \
               --query "Summarise the key findings" \
               --skip-images
```

### Skip both images and tables (fastest, text-only mode)
```bash
python main.py --file data/sample_docs/report.pdf \
               --query "What is the company's strategy?" \
               --skip-images --skip-tables
```

### Interactive Q&A loop
```bash
python main.py --file data/sample_docs/annual_report.pdf --interactive
```

### Use a different model
```bash
python main.py --file data/sample_docs/report.pdf \
               --query "Describe the architecture diagram" \
               --model gpt-4o \
               --vision-model gpt-4o
```

### Full CLI reference
```
--file           Path to PDF document (required)
--query          Question to answer
--model          Text generation model (default: gpt-4)
--vision-model   Vision model for image captioning (default: gpt-4-vision-preview)
--skip-images    Skip GPT-4V image captioning
--skip-tables    Skip LLM table description generation
--interactive    Interactive Q&A loop after indexing
```

---

## Cost Considerations

> ⚠️ **GPT-4V calls cost more.  Use `--skip-images` during development.**

Approximate costs per document (GPT-4V "high" detail, ~1024×1024 images):
- Each image ≈ 765 input tokens ≈ **$0.008–$0.01** at current pricing
- A 50-page document with 20 images ≈ **$0.15–$0.20** in image captioning alone
- Captions are generated once and cached; re-running queries does not re-caption

**Cost optimisation tips:**
1. `--skip-images` — bypass GPT-4V entirely during development
2. Pre-generate captions once, save to JSON, reload on subsequent runs
3. Use LLaVA locally (free) for development, GPT-4V for production
4. Use `gpt-3.5-turbo` for table descriptions (cheaper, still good at structured data)

---

## What This Can vs Cannot Answer

### CAN answer (multimodal RAG)
- "What does the flowchart in section 3 show?" → image caption search
- "What was Q3 revenue according to the table?" → table description search
- "Describe the network architecture diagram" → image caption search
- "What were the year-over-year growth percentages?" → table search
- "Explain the data pipeline shown in figure 2" → image + text combined

### CANNOT answer (text-only RAG)
- "What does the flowchart in section 3 show?" — no image content indexed
- "What was Q3 revenue?" — only if the number also appeared in prose

---

## Comparison with Project 1 (Basic RAG)

| Feature | Project 1 (Basic RAG) | Project 4 (Multimodal RAG) |
|---|---|---|
| Content types | Text only | Text + Images + Tables |
| Indexes | 1 FAISS index | 3 FAISS indexes |
| Embedding model | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 (same) |
| LLM calls | Generation only | Captioning + Table desc + Classification + Generation |
| Query routing | None (always searches) | Router classifies query → selects relevant index(es) |
| Image understanding | ❌ | ✅ GPT-4V captions |
| Table understanding | ❌ | ✅ LLM-generated descriptions |
| Cost | Low (local embeddings) | Medium–High (GPT-4V for images) |
| Complexity | Low | High |

**New concepts introduced in this project:**
- Multimodal document parsing (pdfplumber)
- Vision model integration (GPT-4V via base64 image encoding)
- Multiple specialised FAISS indexes (one per modality)
- Query routing / intent classification
- Cross-modality result merging and ranking
- Modality-aware generation prompts

---

## Project Structure

```
04-multimodal-rag/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── sample_docs/         ← Put your PDF files here
│   └── extracted/
│       ├── images/          ← PNG files extracted from PDFs
│       └── tables/          ← CSV files extracted from PDFs
├── src/
│   ├── multimodal_parser.py ← PDF → text + images + tables
│   ├── text_indexer.py      ← FAISS index for text chunks
│   ├── image_processor.py   ← GPT-4V image captioning
│   ├── image_indexer.py     ← FAISS index for image captions
│   ├── table_processor.py   ← LLM table → prose description
│   ├── table_indexer.py     ← FAISS index for table descriptions
│   ├── query_router.py      ← Classify query → modality(ies)
│   ├── multi_retriever.py   ← Fetch + merge results from indexes
│   └── generator.py         ← Build prompt + call LLM
└── main.py                  ← CLI entry point
```

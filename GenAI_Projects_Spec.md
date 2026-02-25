# GenAI Learning Projects — Specification Guide

**Audience:** Developers/Testers with 3–4 years Java/Python experience, new to Generative AI  
**Goal:** Learn by doing — each project is self-contained, well-commented, and replicable with minimal changes  
**Repository Structure:** Single GitHub repo with one folder per project + a root-level README

---

## Repository Layout

```
genai-beginner-projects/
│
├── README.md                        ← Repo-level overview and navigation guide
│
├── 01-rag-from-scratch/
├── 02-legal-ai-assistant/
├── 03-research-agent/
├── 04-multimodal-rag/
└── 05-agentic-rag-realtime/
```

Each project folder contains:
- `README.md` — what the project does, why it matters, how to run it
- `requirements.txt` — all Python dependencies pinned to versions
- `.env.example` — environment variable template
- `main.py` (or equivalent entry point)
- Well-commented source files organized by feature

---

---

# Project 1: RAG From Scratch

**Folder:** `01-rag-from-scratch/`

## What This Project Teaches

Retrieval-Augmented Generation (RAG) is the foundation of most real-world GenAI applications. Instead of relying on the LLM's training data (which can be outdated or hallucinated), RAG grounds the AI's answers in your own documents. This project teaches you how the pipeline works end-to-end.

## High-Level Flow

```
Your Documents → Chunk & Embed → Store in FAISS
User Question  → Embed Question → Search FAISS → Retrieve Top Chunks → LLM → Answer
```

## Learning Outcomes

- Understand why LLMs hallucinate and how RAG prevents it
- Learn what embeddings are and why semantic search beats keyword search
- Build a document indexing pipeline from scratch
- Wire a retrieval chain using LangChain

## Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | Orchestration framework — connects all pieces |
| FAISS | Vector database — stores and searches embeddings locally |
| HuggingFace Embeddings | Converts text to numerical vectors |
| OpenAI / Ollama | LLM for generating the final answer |
| PyPDF / python-docx | Parsing uploaded documents |

## Folder Structure

```
01-rag-from-scratch/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── sample_docs/          ← Drop your PDFs/text files here
├── src/
│   ├── document_loader.py    ← Step 1: Load and parse documents
│   ├── chunker.py            ← Step 2: Split docs into chunks
│   ├── embedder.py           ← Step 3: Convert chunks to vectors
│   ├── vector_store.py       ← Step 4: Store/load FAISS index
│   ├── retriever.py          ← Step 5: Search for relevant chunks
│   └── generator.py          ← Step 6: Send chunks + question to LLM
└── main.py                   ← Entry point that ties it all together
```

## Feature Breakdown

### document_loader.py
- Load `.pdf`, `.txt`, `.docx` files from the `data/` folder
- Use LangChain's `PyPDFLoader` and `TextLoader`
- **Comment guideline:** Explain what a "Document" object is in LangChain

### chunker.py
- Use `RecursiveCharacterTextSplitter` with `chunk_size=500`, `chunk_overlap=50`
- **Comment guideline:** Explain why overlap matters (context at boundaries)

### embedder.py
- Use `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2` model (free, no API key needed)
- Show what a vector looks like (print shape/first 5 values)
- **Comment guideline:** Explain that similar text produces similar vectors

### vector_store.py
- Create FAISS index from embedded chunks
- Save index to disk (`faiss_index/`) so it doesn't rebuild every run
- Load from disk if it already exists
- **Comment guideline:** Explain FAISS as a similarity search engine

### retriever.py
- Take a user question, embed it, search FAISS for top-k similar chunks
- Default `k=3` with a configurable parameter
- **Comment guideline:** Show the retrieved chunks so learners can see what gets passed to LLM

### generator.py
- Build a LangChain `RetrievalQA` chain
- Use a prompt template that explicitly tells the LLM to only use the provided context
- Show the full prompt being sent (in debug mode)
- **Comment guideline:** Explain the system prompt pattern that prevents hallucination

## README Must Include
- "What is RAG and why does it matter?" — plain English explanation
- Step-by-step setup (create venv, install deps, set `.env`)
- How to add your own documents
- How to run and test with sample questions
- How to verify the LLM is using your documents and not making things up

## Beginner Tips to Include in Comments
- What happens if chunk_size is too large vs. too small
- Why we use cosine similarity vs. exact match
- What `k` means in top-k retrieval

---

---

# Project 2: Legal AI Assistant

**Folder:** `02-legal-ai-assistant/`

## What This Project Teaches

Legal documents are dense, jargon-heavy, and hard for non-lawyers to understand. This project builds an AI assistant that reads contracts and legal docs, summarizes key points, flags risky or conflicting clauses, and answers questions — acting like a junior legal associate.

## High-Level Flow

```
Upload Contract (PDF/DOCX)
        ↓
   Parse & Chunk
        ↓
   Embed & Index (FAISS)
        ↓
   ┌────────────────────────────────┐
   │  Auto-Analysis Pipeline        │
   │  - Executive Summary           │
   │  - Key Clause Extraction       │
   │  - Risk Flags                  │
   │  - Conflict Detection          │
   └────────────────────────────────┘
        ↓
   Interactive Q&A ("Ask the contract anything")
```

## Learning Outcomes

- Build on RAG knowledge from Project 1 with domain-specific prompting
- Learn few-shot prompting to guide LLM output format
- Practice chain-of-thought prompting for legal reasoning
- Learn how to structure a multi-step analysis pipeline

## Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | Chains and prompt management |
| FAISS | Document vector store |
| OpenAI GPT-4 / Claude | Reasoning-capable LLM for legal analysis |
| PyPDF2 / python-docx | Document parsing |
| Rich / Streamlit (optional) | Better terminal or web output |

## Folder Structure

```
02-legal-ai-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── sample_contracts/       ← Sample NDA, service agreement, employment contract
├── src/
│   ├── document_parser.py      ← Parse legal docs and preserve section structure
│   ├── indexer.py              ← Embed and store in FAISS (reuse from Project 1)
│   ├── summarizer.py           ← Generate executive summary
│   ├── clause_extractor.py     ← Extract key clauses (indemnity, liability, termination)
│   ├── risk_analyzer.py        ← Identify risky or one-sided language
│   ├── conflict_detector.py    ← Find contradictions between clauses
│   └── qa_chain.py             ← Interactive Q&A with the document
├── prompts/
│   ├── summary_prompt.txt      ← Prompt template for summarization
│   ├── clause_prompt.txt       ← Prompt template for clause extraction
│   └── risk_prompt.txt         ← Prompt template for risk analysis
└── main.py
```

## Feature Breakdown

### document_parser.py
- Parse PDFs and DOCX files, preserving section headings and numbering
- Identify and label major sections (parties, definitions, terms, obligations)
- **Comment guideline:** Explain why preserving structure matters for legal docs

### summarizer.py
- Generate a plain-English executive summary (max 300 words)
- Identify: parties involved, contract type, duration, key obligations
- Use a structured output prompt: `{"parties": ..., "type": ..., "duration": ..., "summary": ...}`
- **Comment guideline:** Show the before/after — raw clause vs. plain English summary

### clause_extractor.py
- Extract specific clause types: indemnification, limitation of liability, termination, governing law, IP ownership, confidentiality
- Return each clause with its original text and a plain-English translation
- **Comment guideline:** Explain what each clause type means and why it matters

### risk_analyzer.py
- Flag clauses with risky patterns: unlimited liability, one-sided termination, vague language, auto-renewal traps
- Rate risk as HIGH / MEDIUM / LOW with a reason
- **Comment guideline:** Show example of a high-risk vs. standard clause side by side

### conflict_detector.py
- Compare clauses against each other for contradictions (e.g., two different notice periods)
- Check for clauses that contradict stated definitions
- **Comment guideline:** Explain that this is a known challenge even for LLMs — verify manually

### qa_chain.py
- Allow free-form questions: "What happens if I terminate early?" "Who owns the IP I create?"
- Ground answers in document chunks using RAG
- Add source citation: "Based on Section 4.2: ..."
- **Comment guideline:** Explain why citing sources is critical in legal context

## README Must Include
- Disclaimer: "This is for learning purposes. Not legal advice."
- What document types are supported
- Sample questions to ask the assistant
- Explanation of each analysis output section
- Limitations: what this tool cannot reliably detect

---

---

# Project 3: AI Research Agent

**Folder:** `03-research-agent/`

## What This Project Teaches

This project introduces the concept of AI **agents** — LLMs that don't just answer questions but take sequences of actions to complete a goal. The agent reads multiple research papers, synthesizes understanding across them, identifies gaps in the literature, and proposes future research directions.

## High-Level Flow

```
User provides research topic
        ↓
Agent searches for / accepts research papers (PDF)
        ↓
For each paper:
  - Extract title, abstract, methodology, findings, limitations
        ↓
Cross-paper synthesis:
  - Common themes and agreements
  - Contradictions and open questions
  - Missing experiments or populations
        ↓
Output:
  - Structured literature summary
  - Gap analysis report
  - Suggested research paths (with rationale)
```

## Learning Outcomes

- Understand what an AI Agent is vs. a simple LLM call
- Learn to use LangChain Agents with tools
- Practice multi-step reasoning with chain-of-thought prompting
- Build a real research workflow automation tool

## Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain Agents | Agent framework with tool use |
| FAISS | Paper vector store |
| OpenAI / Anthropic | LLM backbone for reasoning |
| PyPDF2 | Parse academic PDFs |
| ArXiv API (optional) | Fetch papers programmatically |
| Pydantic | Structured output validation |

## Folder Structure

```
03-research-agent/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── papers/               ← Drop research papers (PDF) here
├── src/
│   ├── paper_parser.py       ← Extract structured data from each paper
│   ├── paper_indexer.py      ← Embed and index all papers in FAISS
│   ├── tools/
│   │   ├── search_tool.py    ← Tool: Search indexed papers
│   │   ├── summary_tool.py   ← Tool: Summarize a single paper
│   │   └── compare_tool.py   ← Tool: Compare findings across papers
│   ├── agent.py              ← LangChain Agent definition with tools
│   ├── gap_analyzer.py       ← Synthesize gaps from all paper summaries
│   └── report_generator.py   ← Format final output as structured report
└── main.py
```

## Feature Breakdown

### paper_parser.py
- Extract: title, authors, year, abstract, methodology, key findings, limitations
- Use structured prompt: "Extract the following fields from this paper section..."
- Handle messy PDF formatting gracefully
- **Comment guideline:** Explain why we extract structured fields vs. raw text

### paper_indexer.py
- Index each paper's chunks separately, tagging metadata (paper title, section)
- **Comment guideline:** Explain metadata filtering — "search only within this paper"

### tools/search_tool.py
- LangChain `Tool` that takes a query and returns relevant chunks from any paper
- **Comment guideline:** Explain what a LangChain Tool is and how agents use them

### tools/compare_tool.py
- Given two paper titles, compare their methodologies and findings
- Returns structured comparison
- **Comment guideline:** This is where agents shine — multi-step reasoning across documents

### agent.py
- Define a `ReAct` agent with the above tools
- Agent decides which tool to call and in what order
- Log each agent step so learners can see the reasoning chain
- **Comment guideline:** Explain the ReAct loop: Reason → Act → Observe → Repeat

### gap_analyzer.py
- After individual summaries, run a synthesis prompt across all findings
- Identify: what's been studied, what contradicts, what's missing
- **Comment guideline:** Explain that this is prompted reasoning, not database logic

### report_generator.py
- Output a structured report with sections: Overview, Key Findings, Contradictions, Research Gaps, Suggested Next Steps
- Save as `.txt` or `.md` file
- **Comment guideline:** Show how to use output parsers to enforce structure

## README Must Include
- Explanation of what an AI Agent is (with a simple analogy)
- How to add your own papers
- How to interpret the gap analysis output
- Limitations: LLMs can hallucinate citations — always verify

---

---

# Project 4: Multimodal RAG System

**Folder:** `04-multimodal-rag/`

## What This Project Teaches

Most RAG systems only handle text. Real-world documents contain images, charts, tables, and diagrams. This project builds a RAG system that can understand and retrieve across multiple modalities — text, images, and tables — using different models for each type.

## High-Level Flow

```
Mixed Document (PDF with text + images + tables)
        ↓
   Multimodal Parser
   ├── Text chunks      → Text Embedding Model → FAISS Text Index
   ├── Images           → Vision Model (caption/describe) → FAISS Image Index
   └── Tables           → Table Extraction → Tabular Index
        ↓
   User Query
        ↓
   Query Router → Searches appropriate index (or all)
        ↓
   Combine retrieved context → LLM → Answer
```

## Learning Outcomes

- Understand how different modalities require different models
- Learn to use vision models (GPT-4V, LLaVA) for image understanding
- Build a multi-index retrieval system
- Implement a query router to decide which index to search

## Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | Orchestration |
| FAISS | Separate vector stores per modality |
| OpenAI GPT-4V / LLaVA | Image understanding |
| `unstructured` library | Extract images and tables from PDFs |
| `camelot` / `pdfplumber` | Table extraction |
| `sentence-transformers` | Text and cross-modal embeddings |
| CLIP (optional) | Image-text embedding alignment |

## Folder Structure

```
04-multimodal-rag/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── sample_docs/              ← PDFs with images, tables, charts
│   └── extracted/
│       ├── images/               ← Auto-extracted images
│       └── tables/               ← Auto-extracted tables as CSV
├── src/
│   ├── multimodal_parser.py      ← Separate text, images, tables from PDF
│   ├── text_indexer.py           ← Text chunk embedding (same as Project 1)
│   ├── image_processor.py        ← Caption images using vision model
│   ├── image_indexer.py          ← Embed captions, store with image reference
│   ├── table_processor.py        ← Convert tables to text description
│   ├── table_indexer.py          ← Index table descriptions
│   ├── query_router.py           ← Decide which index(es) to search
│   ├── multi_retriever.py        ← Retrieve from multiple indexes, merge results
│   └── generator.py              ← Generate answer from multi-modal context
└── main.py
```

## Feature Breakdown

### multimodal_parser.py
- Use `unstructured` to extract text blocks, image blocks, and table blocks from PDF
- Save extracted images to `data/extracted/images/` as PNG
- Save tables to `data/extracted/tables/` as CSV
- **Comment guideline:** Explain why we separate modalities before indexing

### image_processor.py
- For each image: send to GPT-4V or LLaVA with prompt: "Describe this image in detail for a search system"
- Store: original image path + generated caption
- **Comment guideline:** Explain that we're converting images to text so we can search them

### table_processor.py
- Convert each table to a natural language description: "This table shows Q1-Q4 revenue by region..."
- Keep the raw CSV for reference
- **Comment guideline:** Explain the challenge of searching tabular data semantically

### query_router.py
- Classify the user query: "Is this a visual question? A data question? A text question?"
- Route to: text index, image index, table index, or all three
- **Comment guideline:** Show the classification prompt and how to parse the output

### multi_retriever.py
- Retrieve top-k from each relevant index
- Merge and de-duplicate results
- Score/rank by relevance
- **Comment guideline:** Explain the challenge of ranking across different modalities

## Model Comparison Table (for README)

| Modality | Model | Why This Model |
|----------|-------|----------------|
| Text | all-MiniLM-L6-v2 | Fast, free, good quality |
| Images | GPT-4V / LLaVA | Understands visual content |
| Tables | text-davinci / GPT-3.5 | Strong at structured data |
| Cross-modal | CLIP | Aligns text and image space |

## README Must Include
- Visual diagram of the multi-index architecture
- Explanation of what "multimodal" means with examples
- Comparison: what this can answer that text-only RAG cannot
- Cost considerations (vision model APIs are more expensive)

---

---

# Project 5: Agentic RAG with Real-Time Data

**Folder:** `05-agentic-rag-realtime/`

## What This Project Teaches

Standard RAG uses static documents. But many questions require **live data** — current stock prices, today's news, real-time weather, or up-to-date API data. This project combines RAG (your own knowledge base) with agentic tool use (live data fetching) so the AI can answer questions that span both stored knowledge and the real world.

## High-Level Flow

```
User Question
      ↓
   Agent Planner (decides what it needs)
      ↓
   ┌──────────────────────────────────────────┐
   │ Tool Selection (agent picks what to use) │
   │                                          │
   │  Tool A: RAG Search (your documents)     │
   │  Tool B: Web Search (live news/info)     │
   │  Tool C: Financial API (stock data)      │
   │  Tool D: Weather API (current conditions)│
   │  Tool E: Wikipedia (background context)  │
   └──────────────────────────────────────────┘
      ↓
   Combine results from multiple tools
      ↓
   LLM synthesizes final answer with citations
```

## Learning Outcomes

- Understand Agentic RAG vs. standard RAG
- Learn to build and register custom LangChain Tools
- Understand how agents plan multi-step tool use
- Learn to handle API calls, rate limits, and errors gracefully

## Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain Agents | ReAct agent framework |
| FAISS | Local knowledge base |
| SerpAPI / Tavily | Live web search |
| yfinance | Real-time stock/financial data |
| OpenWeatherMap API | Real-time weather |
| Wikipedia API | Background knowledge |
| OpenAI / Anthropic | LLM for reasoning and synthesis |

## Folder Structure

```
05-agentic-rag-realtime/
├── README.md
├── requirements.txt
├── .env.example              ← API keys for all tools
├── data/
│   └── knowledge_base/       ← Your static documents (RAG source)
├── src/
│   ├── knowledge_indexer.py  ← Index your documents in FAISS
│   ├── tools/
│   │   ├── rag_tool.py       ← Tool: Search local knowledge base
│   │   ├── web_search_tool.py← Tool: Search the live web
│   │   ├── finance_tool.py   ← Tool: Get stock/financial data
│   │   ├── weather_tool.py   ← Tool: Get current weather
│   │   └── wiki_tool.py      ← Tool: Search Wikipedia
│   ├── tool_registry.py      ← Register all tools for the agent
│   ├── agent.py              ← Agent definition with tool list
│   └── response_formatter.py ← Format answer with sources cited
└── main.py
```

## Feature Breakdown

### tools/rag_tool.py
- Wraps the FAISS retriever as a LangChain Tool
- Description: "Use this to answer questions about [your domain]. Input: a search query."
- **Comment guideline:** The tool description is critical — the agent reads it to decide when to use this tool

### tools/web_search_tool.py
- Use SerpAPI or Tavily to fetch live search results
- Return top 3 results with title, snippet, and URL
- **Comment guideline:** Explain rate limits and why you shouldn't call this for every question

### tools/finance_tool.py
- Use `yfinance` to fetch: current price, 52-week high/low, P/E ratio
- Parse the user query to extract ticker symbol
- **Comment guideline:** Show how to handle cases where the ticker isn't found

### tools/weather_tool.py
- Use OpenWeatherMap free API
- Return: temperature, conditions, humidity, forecast
- **Comment guideline:** Explain how agents pass parameters to tools

### tool_registry.py
- Register all tools in a list with names and descriptions
- **Comment guideline:** Explain that the agent's LLM reads these descriptions to make decisions

### agent.py
- Use `initialize_agent` with `AgentType.OPENAI_FUNCTIONS` or `ZERO_SHOT_REACT_DESCRIPTION`
- Enable verbose mode so learners see every reasoning step
- Add memory so the agent can handle follow-up questions
- **Comment guideline:** Print the agent's thought process at each step — this is the key learning

### response_formatter.py
- Format the final answer with: Answer, Tools Used, Sources
- Example:
  ```
  Answer: The stock is currently at $182.50, up 2.3% today. Based on our internal
          analysis documents, this aligns with the Q3 forecast.

  Tools Used: finance_tool, rag_tool
  Sources: Yahoo Finance (live), Q3_forecast.pdf (internal)
  ```
- **Comment guideline:** Explain why showing sources builds trust and helps verify accuracy

## Example Queries This System Can Handle

- "What is the current price of AAPL and how does it compare to our internal valuation model?"
- "What's the weather in London today, and should we proceed with our outdoor event based on our planning guidelines?"
- "What are the latest AI news stories and how do they relate to our company's AI strategy document?"

## README Must Include
- Architecture diagram showing agent + tools
- API key setup guide for each service (with free tier links)
- Explanation of how the agent decides which tool to use
- Examples of multi-tool queries
- Cost and rate limit considerations per tool
- How to add a custom tool (template + walkthrough)

---

---

# Repository-Level README

## Contents

The root `README.md` should include:

1. **Introduction** — Why these 5 projects, what you'll learn, suggested order
2. **Prerequisites** — Python 3.10+, OpenAI API key (or Ollama for free), Git
3. **Project Map** — Table linking each project folder with a one-line description and difficulty rating
4. **Learning Path** — Recommended order (1 → 2 → 3 → 4 → 5) with skill progression
5. **Setup** — How to clone the repo, set up a virtual environment
6. **Glossary** — Plain-English definitions: RAG, embedding, vector store, agent, tool, chain, prompt template, hallucination

## Suggested Difficulty Ratings

| Project | Difficulty | Key New Concept |
|---------|-----------|----------------|
| 1 — RAG From Scratch | ⭐⭐ Beginner | Embeddings, vector search |
| 2 — Legal AI | ⭐⭐⭐ Beginner+ | Domain prompting, structured output |
| 3 — Research Agent | ⭐⭐⭐ Intermediate | Agents, multi-step reasoning |
| 4 — Multimodal RAG | ⭐⭐⭐⭐ Intermediate | Vision models, multi-index |
| 5 — Agentic RAG + Real-Time | ⭐⭐⭐⭐ Intermediate | Tool use, live data APIs |

---

*This specification is intended as a blueprint. Implementation details (exact library versions, API providers) can be substituted based on availability and cost preference. All projects should work with both OpenAI API and local Ollama models where possible, so learners aren't gated by cost.*

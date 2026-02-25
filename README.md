# GenAI Beginner Projects

A hands-on collection of **five Generative AI projects** designed for developers and testers with 3–4 years of Java/Python experience who are new to Generative AI. Each project is self-contained, well-commented, and can be replicated with minimal changes.

> **Learn by doing** — start with a simple RAG pipeline and progressively build up to agentic systems with real-time data.

---

## Prerequisites

- **Python 3.10+** — [Download here](https://www.python.org/downloads/)
- **OpenAI API key** (or [Ollama](https://ollama.ai/) for free local models)
- **Git** — for cloning the repository
- Basic familiarity with Python, pip, and virtual environments

---

## Project Map

| # | Project | Folder | Difficulty | Key New Concept |
|---|---------|--------|-----------|----------------|
| 1 | RAG From Scratch | [`01-rag-from-scratch/`](./01-rag-from-scratch/) | ⭐⭐ Beginner | Embeddings, vector search |
| 2 | Legal AI Assistant | [`02-legal-ai-assistant/`](./02-legal-ai-assistant/) | ⭐⭐⭐ Beginner+ | Domain prompting, structured output |
| 3 | AI Research Agent | [`03-research-agent/`](./03-research-agent/) | ⭐⭐⭐ Intermediate | Agents, multi-step reasoning |
| 4 | Multimodal RAG | [`04-multimodal-rag/`](./04-multimodal-rag/) | ⭐⭐⭐⭐ Intermediate | Vision models, multi-index |
| 5 | Agentic RAG + Real-Time | [`05-agentic-rag-realtime/`](./05-agentic-rag-realtime/) | ⭐⭐⭐⭐ Intermediate | Tool use, live data APIs |

---

## Learning Path

We recommend completing the projects in order. Each builds on concepts from the previous one:

```
Project 1: RAG From Scratch
   └─ Learn embeddings, vector stores, and retrieval chains
        │
        ▼
Project 2: Legal AI Assistant
   └─ Apply RAG to a domain; learn structured output and few-shot prompting
        │
        ▼
Project 3: AI Research Agent
   └─ Introduce agents that take actions and use tools autonomously
        │
        ▼
Project 4: Multimodal RAG
   └─ Handle images, tables, and text in a single pipeline
        │
        ▼
Project 5: Agentic RAG + Real-Time Data
   └─ Combine RAG with live APIs; build a full agentic system
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/nerdjerry/rag-projects.git
cd rag-projects
```

### 2. Set up a virtual environment (per project)

```bash
cd 01-rag-from-scratch   # or any project folder
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run the project

```bash
python main.py
```

Each project README has detailed, project-specific setup instructions.

---

## Glossary

| Term | Plain-English Definition |
|------|--------------------------|
| **RAG** (Retrieval-Augmented Generation) | A pattern where the AI retrieves relevant documents before generating an answer, so it uses your data instead of guessing. |
| **Embedding** | A list of numbers (a vector) that represents the meaning of a piece of text. Similar texts produce similar vectors. |
| **Vector Store** | A database optimized for storing and searching embeddings. Think of it as a search engine for meaning, not keywords. |
| **Agent** | An AI that doesn't just answer — it plans, picks tools, takes actions, and iterates until it reaches a goal. |
| **Tool** | A function an agent can call — like searching the web, querying a database, or fetching live stock prices. |
| **Chain** | A sequence of steps (e.g., retrieve → format → generate) connected together in a pipeline. |
| **Prompt Template** | A reusable text pattern with placeholders (like `{question}`) that gets filled in at runtime before being sent to the LLM. |
| **Hallucination** | When an LLM confidently generates information that is incorrect or fabricated. RAG reduces this by grounding answers in real documents. |
| **Chunk** | A small piece of a larger document. We split documents into chunks because LLMs have limited context windows. |
| **Top-k Retrieval** | Finding the `k` most relevant chunks for a query. Higher `k` = more context but more noise. |

---

## Repository Structure

```
rag-projects/
│
├── README.md                        ← You are here
├── GenAI_Projects_Spec.md           ← Full specification document
│
├── 01-rag-from-scratch/             ← Project 1: Build a RAG pipeline
├── 02-legal-ai-assistant/           ← Project 2: Legal document analyzer
├── 03-research-agent/               ← Project 3: Research paper agent
├── 04-multimodal-rag/               ← Project 4: Text + image + table RAG
└── 05-agentic-rag-realtime/         ← Project 5: Agent with live data tools
```

---

*All projects work with both OpenAI API and local Ollama models where possible, so learners aren't gated by cost.*

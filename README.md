# GenAI Beginner Projects

A hands-on learning path for developers new to Generative AI. Five self-contained projects that take you from basic RAG to agentic systems with real-time data — each building on the previous.

---

## Why These 5 Projects?

Most GenAI tutorials show you a hello-world demo and call it a day. These projects are different:

- **Real code**, not toy examples — each project solves an actual use case
- **Step-by-step comments** explain *why*, not just *what*
- **Progressive complexity** — each project introduces exactly one new concept
- **Works with OpenAI or Ollama** — you're not gated by API costs

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | `python --version` to check |
| OpenAI API key | Or run Ollama locally for free |
| Git | For cloning the repo |
| 8 GB RAM minimum | For running local embedding models |

---

## Project Map

| # | Project | Difficulty | Key New Concept | One-Line Description |
|---|---------|-----------|----------------|----------------------|
| 1 | [RAG From Scratch](./01-rag-from-scratch/) | ⭐⭐ Beginner | Embeddings, vector search | Build a Q&A system over your own documents |
| 2 | [Legal AI Assistant](./02-legal-ai-assistant/) | ⭐⭐⭐ Beginner+ | Domain prompting, structured output | Analyze contracts for risks, clauses, and conflicts |
| 3 | [AI Research Agent](./03-research-agent/) | ⭐⭐⭐ Intermediate | Agents, multi-step reasoning | Synthesize multiple research papers and find gaps |
| 4 | [Multimodal RAG](./04-multimodal-rag/) | ⭐⭐⭐⭐ Intermediate | Vision models, multi-index | RAG that understands text, images, and tables |
| 5 | [Agentic RAG + Real-Time](./05-agentic-rag-realtime/) | ⭐⭐⭐⭐ Intermediate | Tool use, live data APIs | Agent that combines stored docs with live web/financial data |

---

## Learning Path

Follow the projects in order — each one adds exactly one new layer:

```
Project 1: RAG From Scratch
    ↓  (adds domain-specific prompting)
Project 2: Legal AI Assistant
    ↓  (adds agent framework + tools)
Project 3: AI Research Agent
    ↓  (adds vision models + multi-index)
Project 4: Multimodal RAG
    ↓  (adds live data tools + planning)
Project 5: Agentic RAG + Real-Time
```

**Skill progression:**
- After Project 1: You understand how RAG works and can build basic Q&A over documents
- After Project 2: You can write domain-specific prompts and structure LLM output as JSON
- After Project 3: You understand agents and can build multi-step reasoning systems
- After Project 4: You can handle documents with images and tables, not just text
- After Project 5: You can build production-grade agents that combine stored knowledge with live data

---

## Quick Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-org/genai-beginner-projects.git
cd genai-beginner-projects

# 2. Pick a project to start with
cd 01-rag-from-scratch

# 3. Create a virtual environment (recommended — keeps dependencies isolated)
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 6. Run the project
python main.py --help
```

> **Tip:** Each project has its own `venv` and `requirements.txt`. You don't need to install everything at once.

---

## Glossary

Plain-English definitions for terms you'll encounter in these projects:

| Term | Plain-English Definition |
|------|--------------------------|
| **RAG** | Retrieval-Augmented Generation — feeding relevant documents to an LLM before asking it a question, so it answers based on your data instead of guessing |
| **Embedding** | A list of numbers (a vector) that represents the meaning of a piece of text. Similar texts have similar vectors. |
| **Vector store** | A database optimized for finding similar vectors quickly. FAISS is a popular local option. |
| **FAISS** | Facebook AI Similarity Search — a library that stores vectors and finds the most similar ones very fast |
| **Agent** | An LLM that can take actions (like calling tools) to complete a goal, rather than just answering a single question |
| **Tool** | A function the agent can call — like searching the web, getting stock prices, or searching your documents |
| **Chain** | A sequence of LLM calls or operations linked together. LangChain helps you build these. |
| **Prompt template** | A reusable text structure with variables that gets filled in at runtime. Like a form letter for LLMs. |
| **Hallucination** | When an LLM confidently states something false. RAG reduces this by grounding answers in real documents. |
| **Chunk** | A small piece of a larger document (usually 300–1000 characters). Documents are split into chunks for embedding. |
| **Top-k retrieval** | Finding the k most similar chunks to a question. k=3 means: "find the 3 most relevant passages." |
| **ReAct** | Reason + Act — an agent pattern where the LLM thinks about what to do, does it, observes the result, and repeats |
| **LangChain** | A Python framework for building LLM applications. Provides building blocks for RAG, agents, chains, and more. |

---

## Using Ollama (Free Local LLMs)

Don't want to pay for OpenAI? Use Ollama to run LLMs on your own machine:

```bash
# Install Ollama: https://ollama.com
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# In any project, use:
python main.py --model ollama/llama3
```

> **Note:** Local models require ~8 GB RAM for small models. They're slower than OpenAI but completely free.

---

## Repository Structure

```
genai-beginner-projects/
│
├── README.md                        ← You are here
│
├── 01-rag-from-scratch/             ← ⭐⭐ Build RAG from scratch
├── 02-legal-ai-assistant/           ← ⭐⭐⭐ Legal document analysis
├── 03-research-agent/               ← ⭐⭐⭐ AI research synthesis agent
├── 04-multimodal-rag/               ← ⭐⭐⭐⭐ Text + images + tables
└── 05-agentic-rag-realtime/         ← ⭐⭐⭐⭐ Live data + documents
```

Each project folder contains:
- `README.md` — what the project does, how to run it, what you'll learn
- `requirements.txt` — all Python dependencies pinned to specific versions
- `.env.example` — copy this to `.env` and fill in your API keys
- `main.py` — the entry point to run the project
- `src/` — well-commented source files organized by feature

---

## Contributing

Found a bug? Have an improvement? Open an issue or PR.

When adding comments or documentation, remember the audience: developers with 3–4 years of experience who are new to GenAI. Explain the "why", not just the "what".

---

*All projects support both OpenAI API and local Ollama models. You don't need to pay for API access to learn from these projects.*

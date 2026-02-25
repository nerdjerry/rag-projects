# 🔬 Project 3: AI Research Agent

An AI-powered research assistant that reads academic papers, indexes them for
search, and uses an autonomous **agent** to answer questions, compare studies,
and identify research gaps.

---

## 🤔 What Is an AI Agent?

**Simple analogy:** Imagine you hire a research assistant. You don't give them
a step-by-step script — you give them a question ("What are the gaps in
current NLP research?") and they *figure out* what to do: read papers, take
notes, compare findings, and write a report.

An **AI Agent** works the same way. It has:

| Component   | What it does                                         |
|-------------|------------------------------------------------------|
| **Brain**   | An LLM that reasons about what to do next            |
| **Tools**   | Functions the agent can call (search, summarise, …)  |
| **Memory**  | The conversation/context so far                      |
| **Loop**    | Reason → Act → Observe → Repeat until done           |

The agent in this project follows the **ReAct** pattern:
1. **Reason** — "I need to find the methodology of Paper X."
2. **Act** — Call the `search_papers` tool.
3. **Observe** — Read the search results.
4. **Repeat** — Decide if more info is needed, or produce a final answer.

---

## 🏗️ High-Level Flow

```
┌────────────┐     ┌──────────────┐     ┌──────────────┐
│  PDF Papers│────▶│ Paper Parser  │────▶│ Parsed Data  │
│  (data/)   │     │ (LLM extract)│     │ (structured) │
└────────────┘     └──────────────┘     └──────┬───────┘
                                               │
                   ┌──────────────┐            │
                   │ Paper Indexer│◀───────────┘
                   │ (FAISS)     │
                   └──────┬───────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    ReAct Agent        │
              │  ┌─────────────────┐  │
              │  │ Search Tool     │  │
              │  │ Summary Tool    │  │
              │  │ Compare Tool    │  │
              │  └─────────────────┘  │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   Gap Analyzer        │
              │  (cross-paper synth.) │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Report Generator     │
              │  (Markdown output)    │
              └───────────────────────┘
```

---

## 🧰 Tech Stack

| Technology              | Purpose                                    |
|-------------------------|--------------------------------------------|
| **LangChain**           | Agent framework, prompt templates, tools   |
| **FAISS**               | Vector similarity search over paper chunks |
| **sentence-transformers**| Local embeddings (no API cost)            |
| **pypdf**               | PDF text extraction                        |
| **Pydantic**            | Structured output validation               |
| **OpenAI / Ollama**     | LLM backend (cloud or local)               |

---

## 🚀 Setup Instructions

### 1. Install dependencies

```bash
cd 03-research-agent
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your API key:
```
OPENAI_API_KEY=sk-your-key-here
```

**Or use Ollama (free, local):**
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
```
Then set in `.env`:
```
USE_OLLAMA=true
OLLAMA_MODEL=llama3
```

### 3. Add papers

Place PDF research papers in `data/papers/`:
```
data/papers/
  ├── attention-is-all-you-need.pdf
  ├── bert-pre-training.pdf
  └── gpt4-technical-report.pdf
```

### 4. Run the agent

```bash
python main.py
```

This will:
1. Parse all papers (extract title, methods, findings, etc.)
2. Build a searchable vector index
3. Run a gap analysis across all papers
4. Save a report to `data/report.md`
5. Start an interactive Q&A session

For interactive-only mode (skip the full analysis):
```bash
python main.py --interactive
```

---

## 📖 How to Add Your Own Papers

1. Drop PDF files into `data/papers/`.
2. Delete `data/index/` (to force re-indexing):
   ```bash
   rm -rf data/index/
   ```
3. Run `python main.py` again.

**Tips:**
- Papers should be machine-readable PDFs (not scanned images).
- The parser works best with standard academic paper formatting.
- You can mix papers from different fields — the gap analysis will note
  the diversity.

---

## 🔍 How to Interpret Gap Analysis Output

The gap analysis report contains five sections:

| Section               | What it tells you                                |
|-----------------------|--------------------------------------------------|
| **Well-studied areas**| Topics multiple papers cover — the "known knowns"|
| **Key agreements**    | Findings that papers converge on                 |
| **Contradictions**    | Areas where papers disagree — needs resolution   |
| **Research gaps**     | Unanswered questions — the "known unknowns"      |
| **Suggested next steps** | Promising directions for future research      |

**Example interpretation:**
> "Papers A and B both find that transformer models outperform RNNs on
> sequence tasks (agreement), but they disagree on whether pre-training
> data size matters more than model size (contradiction). No paper
> examines the impact on low-resource languages (gap)."

---

## ⚠️ Limitations

1. **LLMs can hallucinate citations.** The agent may reference findings
   that don't exist in the papers, or misattribute a finding to the wrong
   paper. **Always verify claims against the original PDFs.**

2. **PDF quality matters.** Scanned PDFs, complex layouts, or heavy use
   of figures/tables may result in poor text extraction.

3. **Context window limits.** Very long papers may be truncated before
   the LLM sees them. Key info at the end of a paper might be missed.

4. **No real-time data.** The agent only knows about the papers you provide —
   it cannot search the internet or access live databases.

5. **Embedding model scope.** The local embeddings model is general-purpose.
   Highly specialised domains (e.g. chemistry notation) may benefit from
   domain-specific embeddings.

---

## 📁 Project Structure

```
03-research-agent/
├── main.py                  # Entry point — full pipeline + interactive mode
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── README.md                # This file
├── data/
│   └── papers/              # Place your PDF papers here
├── src/
│   ├── __init__.py
│   ├── paper_parser.py      # Extracts structured fields from PDFs
│   ├── paper_indexer.py     # Chunks and indexes papers in FAISS
│   ├── agent.py             # ReAct agent definition
│   ├── gap_analyzer.py      # Cross-paper synthesis and gap detection
│   ├── report_generator.py  # Markdown report output
│   └── tools/
│       ├── __init__.py
│       ├── search_tool.py   # Vector search over indexed papers
│       ├── summary_tool.py  # Single-paper summarisation
│       └── compare_tool.py  # Side-by-side paper comparison
```

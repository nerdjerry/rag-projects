# 03 — Research Agent

> **"Like a research assistant who can look up papers, take notes, and compare findings — rather than just answering one question."**

## What is an AI Agent?

A regular LLM call is a single prompt → single response.  You hand the model some text and it writes back.  That's it.

An **AI agent** is different: it has access to **tools** — functions it can call to look things up, compute things, or take actions — and it decides *dynamically* which tools to use based on each new sub-goal.

Think of the difference between:
- 🤖 **Simple LLM**: You ask "what does Paper A say about transformers?" and the model guesses from its training data.
- 🕵️ **Research Agent**: You ask the same question, and the agent *looks it up*, reads the relevant sections, possibly *compares* them to Paper B, and synthesises an answer with citations.

## How This Differs from Simple RAG

| Simple RAG | Research Agent |
|---|---|
| Embed documents → vector DB | Same |
| User query → nearest chunks → LLM answer | **Agent plans which tools to call** |
| Single retrieval step | **Multi-step: search → summarise → compare** |
| No memory between steps | **Observations from each step feed the next** |
| Good for Q&A | Good for synthesis, comparison, gap analysis |

In simple RAG, the pipeline is fixed: retrieve then answer.  In an agent, the LLM itself decides the pipeline at runtime.

## The ReAct Loop Explained

**ReAct = Reason + Act**.  The agent alternates between thinking and doing:

```
Thought : I need to find papers about attention mechanisms.
Action  : search_papers
Input   : attention mechanism self-attention
Observation: [Result 1] Paper: "Attention Is All You Need" …

Thought : I found the relevant paper. Now I'll get its full summary.
Action  : summarize_paper
Input   : Attention Is All You Need
Observation: Title: Attention Is All You Need, Authors: Vaswani et al. …

Thought : I have enough to answer the question.
Final Answer: The paper "Attention Is All You Need" introduced …
```

Each **Observation** is the tool's output, appended to the agent's context.  The agent re-reads the growing context at each step to decide what to do next.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    main.py                          │
│  (CLI: --query / --report / --interactive)          │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │    Research Agent       │  ← agent.py
          │  (ReAct loop + LLM)     │
          └──┬──────────┬───────────┘
             │          │
    ┌─────────▼──┐  ┌───▼────────────┐  ┌─────────────────┐
    │search_tool │  │ summary_tool   │  │  compare_tool   │
    │(FAISS      │  │ (PaperMetadata │  │  (LLM comparison│
    │ semantic   │  │  lookup)       │  │   of two papers)│
    │ search)    │  └───────┬────────┘  └────────┬────────┘
    └─────┬──────┘          │                    │
          │         ┌───────▼────────────────────▼──────┐
          │         │        PaperMetadata objects       │
          │         │        (from paper_parser.py)      │
          │         └───────────────────────────────────┘
    ┌─────▼──────┐
    │ FAISS index│  ← paper_indexer.py
    │ (chunked   │
    │  PDFs +    │
    │  metadata) │
    └────────────┘

  Gap Analysis (--report):
  paper_metadata → gap_analyzer.py → LLM synthesis → report_generator.py → .md file
```

## Setup

```bash
# 1. Clone / navigate to the project
cd 03-research-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Add research papers
# Copy your .pdf files into data/papers/
```

## How to Add Papers

Place any number of **.pdf** files into `data/papers/`.  The pipeline will:
1. Extract text and LLM-parse metadata (title, authors, abstract, methodology, findings, limitations).
2. Chunk the full text and embed it into a FAISS vector index.
3. Make both the metadata and the full text available to the agent's tools.

**Tips:**
- Use papers that are topically related for better gap analysis.
- 3–10 papers is the sweet spot.  More than 20 may hit the LLM's context limit during gap analysis.
- Scanned PDFs without OCR will produce empty or garbled text — use PDFs with selectable text.

## Running the Agent

```bash
# Ask a single question and exit
python main.py --query "What methodologies are used across these papers?"

# Start an interactive Q&A session
python main.py --interactive

# Generate a gap analysis report
python main.py --topic "transformer models" --report

# All options
python main.py --papers-dir data/papers \
               --topic "BERT fine-tuning" \
               --model gpt-4 \
               --report \
               --output reports/bert_gaps.md
```

## Sample Queries

These questions showcase the agent's multi-step reasoning:

```
"What methodologies are used across these papers?"
"Which papers agree on X, and which contradict each other?"
"What are the main gaps in this research area?"
"Summarise the paper on [topic] and compare it to [other paper]."
"Which paper has the strongest experimental design?"
"What datasets are most commonly used?"
"Are there any contradictions between the papers' findings?"
```

## How to Interpret the Gap Analysis

The gap analysis report has six sections:

| Section | What it means |
|---|---|
| **Common Themes** | Topics / findings that appear in multiple papers — the consensus view |
| **Contradictions** | Where papers disagree — potential areas of ongoing debate |
| **Missing Experiments** | Experiments that logically follow from the existing work but haven't been done |
| **Missing Populations** | Groups, languages, contexts, or demographics not yet studied |
| **Methodological Gaps** | Approaches not used in any paper (e.g., "no longitudinal study exists") |
| **Suggested Next Steps** | Concrete research directions derived from all of the above |

> ⚠️ **Always verify the output.**  LLMs can hallucinate contradictions or invent plausible-sounding but non-existent gaps.  Treat the gap analysis as a *first draft* to refine with domain expertise.

## Limitations

1. **LLMs can hallucinate citations** — the agent might confidently say "Paper X found Y" when it did not.  Always check claims against the original PDF.

2. **Gap analysis may miss domain-specific context** — a gap that is obvious to a domain expert ("nobody used technique Z") requires domain knowledge the LLM may not have.

3. **Works best with 3–10 papers on the same topic** — fewer papers means less to synthesise; more papers risks exceeding the context window during gap analysis.

4. **PDF extraction quality varies** — scanned PDFs, multi-column layouts, and heavy use of figures degrade text extraction.  The LLM falls back gracefully but metadata may be incomplete.

5. **The agent may loop or over-call tools** — the `max_iterations=8` safety cap prevents infinite loops but may cut off complex multi-paper comparisons.

## How to Extend

### Adding a new tool

1. Create `src/tools/my_tool.py` with a `create_my_tool(…) -> Tool` function.
2. Import and instantiate it in `src/agent.py` inside `create_research_agent`.
3. Add it to the `tools` list passed to `initialize_agent`.

The agent will automatically start using the new tool based on its description — no other changes needed.

### Ideas for new tools

- **`cite_tool`** — generate a BibTeX entry for a paper from its metadata.
- **`timeline_tool`** — order papers chronologically and show how the field evolved.
- **`keyword_tool`** — extract and rank keywords across all papers.
- **`arxiv_tool`** — search arXiv for papers related to the indexed collection.

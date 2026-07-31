# Legal AI Assistant

> ⚠️ **DISCLAIMER: This tool is for educational purposes only. It does NOT constitute legal advice. Always consult a qualified attorney before making any legal or business decisions.**

A Retrieval-Augmented Generation (RAG) pipeline that helps you understand contracts by extracting key clauses, flagging risks, detecting internal conflicts, and answering natural-language questions — all grounded in the actual document text.

---

## What the Tool Does

- **Parses** PDF and DOCX contract files into structured text with section detection
- **Indexes** the document into a FAISS vector store for semantic search
- **Summarises** the contract: parties, type, effective date, duration, key obligations
- **Extracts clauses**: indemnification, limitation of liability, termination, governing law, IP ownership, confidentiality
- **Analyses risks**: flags HIGH / MEDIUM / LOW risk patterns with plain-English explanations
- **Detects conflicts**: surfaces internal contradictions between clauses
- **Answers questions**: grounded Q&A with mandatory section citations

---

## Supported Document Types

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF    | `.pdf`    | Text-based PDFs only. Scanned PDFs require OCR pre-processing. |
| Word   | `.docx`   | Supports Heading styles for better section detection. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                            │
└──────────────┬──────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  document_parser.py  │  PDF / DOCX → full_text + sections
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │     indexer.py       │  text → chunks → HuggingFace embeddings → FAISS
    └──────────┬──────────┘
               │
       ┌───────┴────────┐
       │  OpenAI GPT-4  │  (all LLM calls below use this)
       └───────┬────────┘
               │
    ┌──────────▼──────────┐   ┌──────────────────────┐
    │    summarizer.py     │   │  clause_extractor.py  │
    └─────────────────────┘   └──────────┬───────────┘
                                          │
                               ┌──────────▼──────────┐
                               │   risk_analyzer.py   │
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │ conflict_detector.py │
                               └─────────────────────┘
    ┌─────────────────────┐
    │     qa_chain.py      │  FAISS retriever + custom legal prompt
    └─────────────────────┘
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | `python --version` to check |
| OpenAI API key | Required — this project has no key-free/local-model path |
| A contract PDF or DOCX | `data/sample_contracts/` ships **empty** (just a `.gitkeep`); bring your own file or use any sample service agreement / NDA you have on hand |

---

## Setup

### 1. Clone / navigate to the project

```bash
cd 02-legal-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4          # or gpt-3.5-turbo for lower cost
```

### 5. Add a contract file

`data/sample_contracts/` ships empty — place a PDF or DOCX contract there (or point `--file` at any other path). Don't have one handy? Create a minimal test DOCX in Python:

```bash
python -c "
from docx import Document
d = Document()
d.add_paragraph('SERVICE AGREEMENT between Acme Corp and Beta LLC.')
d.add_paragraph('Section 1. Termination. Either party may terminate with 30 days written notice.')
d.add_paragraph('Section 2. Liability. Total liability is capped at fees paid in the preceding 12 months.')
d.save('data/sample_contracts/test_contract.docx')
"
```

---

## How to Run

### Full analysis (default)

```bash
python main.py --file data/sample_contracts/service_agreement.pdf
```

### Use a cheaper model (faster, less accurate)

```bash
python main.py --file contract.pdf --model gpt-3.5-turbo
```

### Skip risk analysis and conflict detection

```bash
python main.py --file contract.pdf --skip-risks --skip-conflicts
```

### Ask a single question and exit

```bash
python main.py --file contract.pdf --question "What are my termination rights?"
```

### Interactive Q&A after analysis

```bash
python main.py --file contract.pdf --interactive
```

---

## Sample Questions to Ask

```
What are my termination rights?
Who owns IP I create during the contract?
What is the liability cap?
How does auto-renewal work?
What information must I keep confidential and for how long?
Which court has jurisdiction over disputes?
Can the company change the terms without my consent?
What happens to my work if the contract is terminated early?
```

---

## Output Sections Explained

| Section | What it shows |
|---------|---------------|
| **Executive Summary** | Parties, contract type, effective date, duration, key obligations, plain-English overview |
| **Key Clauses** | Table of named clause types with their section references and plain-English translations |
| **Risk Analysis** | 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW risks with explanations and fair alternatives |
| **Conflict Detection** | Internal contradictions between clauses (e.g. mismatched notice periods) |
| **Q&A** | Grounded answers with mandatory section citations |

---

## Limitations

> These are not bugs — they are inherent limitations of the technology.

1. **Cannot reliably detect all conflicts.** The LLM may miss conflicts requiring deep legal expertise or flag false positives. Every flagged conflict must be manually verified.

2. **PDF extraction may miss some formatting.** Tables lose column alignment, scanned PDFs produce no text, and footnotes may appear mid-sentence. Complex formatting in PDFs will degrade extraction quality.

3. **LLM can misinterpret complex legal language.** Highly technical, jurisdiction-specific, or archaic legal terms may be interpreted incorrectly. The model is not a lawyer.

4. **Context window limits truncate long contracts.** Summary and clause extraction are capped at 8 000–12 000 characters. Very long contracts (100+ pages) will have their later sections underweighted.

5. **Embeddings may not capture domain-specific meaning.** The `all-MiniLM-L6-v2` model was not trained on legal text specifically; niche legal terms may not retrieve optimally.

6. **Always verify with a qualified attorney.** This tool helps you know WHAT to look for and WHERE to look. It does not replace professional legal review.

---

## Project Structure

```
02-legal-ai-assistant/
├── README.md                  ← this file
├── requirements.txt
├── .env.example
├── data/
│   └── sample_contracts/      ← place your PDF/DOCX files here
├── src/
│   ├── __init__.py
│   ├── document_parser.py     ← PDF/DOCX → structured text
│   ├── indexer.py             ← text → FAISS vector index
│   ├── summarizer.py          ← executive summary generation
│   ├── clause_extractor.py    ← named clause extraction
│   ├── risk_analyzer.py       ← HIGH/MEDIUM/LOW risk scoring
│   ├── conflict_detector.py   ← internal contradiction detection
│   └── qa_chain.py            ← RAG Q&A chain
├── prompts/
│   ├── summary_prompt.txt
│   ├── clause_prompt.txt
│   └── risk_prompt.txt
└── main.py                    ← CLI entry point
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` + `langchain-community` + `langchain-openai` | LLM orchestration and RAG chains |
| `faiss-cpu` | Local vector similarity search |
| `sentence-transformers` | HuggingFace embedding model (runs locally) |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX parsing |
| `openai` | OpenAI API client |
| `python-dotenv` | `.env` file loading |
| `pydantic` | Data validation |
| `rich` | Formatted terminal output |

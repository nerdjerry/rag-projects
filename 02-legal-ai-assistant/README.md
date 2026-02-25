# Legal AI Assistant

> ⚠️ **DISCLAIMER**: This project is for **educational and learning purposes only**.
> It does **NOT** constitute legal advice. Always consult a qualified attorney for
> legal decisions. AI-generated analysis may contain errors, miss important details,
> or misinterpret legal language.

A RAG-powered tool that analyzes legal contracts and documents using LLMs. Upload a
contract and get a plain-English summary, clause extraction, risk analysis, conflict
detection, and an interactive Q&A interface — all grounded in the actual document text.

---

## Supported Document Types

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF    | `.pdf`    | Text-based PDFs (scanned images require OCR, not supported) |
| Word   | `.docx`   | Microsoft Word documents with paragraph styles |
| Text   | `.txt`    | Plain-text files (fallback parser) |

---

## High-Level Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Upload PDF  │────▶│  Parse &     │────▶│  Build FAISS     │
│  / DOCX /TXT │     │  Structure   │     │  Vector Index    │
└──────────────┘     └──────────────┘     └──────────────────┘
                                                   │
                     ┌─────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │    LLM Analysis        │
        │                        │
        │  ┌──────────────────┐  │
        │  │ Executive Summary│  │
        │  └──────────────────┘  │
        │  ┌──────────────────┐  │
        │  │ Clause Extraction│  │
        │  └──────────────────┘  │
        │  ┌──────────────────┐  │
        │  │ Risk Analysis    │  │
        │  └──────────────────┘  │
        │  ┌──────────────────┐  │
        │  │ Conflict Check   │  │
        │  └──────────────────┘  │
        └────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Interactive Q&A       │
        │  (RAG-grounded)        │
        └────────────────────────┘
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- (Optional) [Ollama](https://ollama.ai/) for free local LLM inference

### 2. Install Dependencies

```bash
cd 02-legal-ai-assistant
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example config
cp .env.example .env

# Edit .env and add your API key (or enable Ollama)
nano .env
```

**Option A — OpenAI (cloud, paid):**
```
OPENAI_API_KEY=sk-your-key-here
USE_OLLAMA=false
```

**Option B — Ollama (local, free):**
```
USE_OLLAMA=true
OLLAMA_MODEL=llama3
```

### 4. Run the Analysis

```bash
python main.py path/to/your/contract.pdf
```

---

## Sample Questions for Interactive Q&A

Once the analysis completes, you can ask follow-up questions like:

- "What is the notice period for termination?"
- "Who owns the intellectual property created under this contract?"
- "Is there a cap on liability? What is the maximum amount?"
- "What happens if one party breaches the agreement?"
- "Which state's laws govern this agreement?"
- "Are there any auto-renewal provisions?"
- "What information is considered confidential?"
- "Can the contract be assigned to a third party?"
- "What are the payment terms?"
- "Is there a non-compete clause?"

---

## Output Sections Explained

### Executive Summary
A ≤300-word plain-English overview of the contract, including who the parties are,
what type of contract it is, key dates, and the main obligations. Designed for busy
stakeholders who need to quickly understand what they're signing.

### Extracted Clauses
Six key clause types are identified and translated:
- **Indemnification** — Who pays if something goes wrong?
- **Limitation of Liability** — Is there a cap on damages?
- **Termination** — How can each party exit the contract?
- **Governing Law** — Which jurisdiction's laws apply?
- **IP Ownership** — Who owns the work product?
- **Confidentiality** — What must be kept secret?

Each clause shows the original legal text alongside a plain-English translation.

### Risk Analysis
Each clause is evaluated for risky patterns and rated:
- **HIGH** — Could result in significant financial or legal exposure
- **MEDIUM** — Creates an imbalance but is somewhat common
- **LOW** — Minor concern, easily fixed in negotiation

Common risks flagged: unlimited liability, one-sided termination, vague language,
auto-renewal traps, overly broad IP assignment, missing protections.

### Conflict Detection
Clauses are compared against each other to find internal contradictions, such as:
- Two sections specifying different governing law
- A definition that conflicts with how the term is used elsewhere
- An obligation in one section negated by a limitation in another

### Interactive Q&A
Ask any question about the contract. Answers are grounded in the actual document
text using RAG (Retrieval-Augmented Generation) and include source citations so
you can verify every claim.

---

## Limitations

1. **Not Legal Advice** — This tool is an educational project. It can miss critical
   issues that a trained lawyer would catch.

2. **LLM Hallucination** — Despite RAG grounding, LLMs can still generate incorrect
   or misleading information. Always verify against the source document.

3. **Scanned PDFs** — Only text-based PDFs are supported. Scanned document images
   would require OCR preprocessing (not included).

4. **Context Window Limits** — Very long contracts (100+ pages) may be truncated
   when sent to the LLM. The tool prioritizes the beginning and end of the document.

5. **Conflict Detection Accuracy** — Finding contradictions between clauses is
   inherently difficult. The tool may flag false positives or miss subtle conflicts.

6. **Language Support** — Currently optimized for English-language contracts only.

7. **No Multi-Document Comparison** — Analyzes one document at a time. Cannot
   compare two versions of a contract or cross-reference multiple agreements.

8. **Embedding Model Limitations** — The default embedding model (`all-MiniLM-L6-v2`)
   is general-purpose, not fine-tuned for legal language. A legal-domain embedding
   model would improve retrieval quality.

---

## Project Structure

```
02-legal-ai-assistant/
├── main.py                     # Entry point — orchestrates the full pipeline
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── README.md                   # This file
├── data/
│   └── sample_contracts/       # Place your contracts here
├── prompts/
│   ├── summary_prompt.txt      # Prompt for executive summary
│   ├── clause_prompt.txt       # Prompt for clause extraction
│   └── risk_prompt.txt         # Prompt for risk analysis
└── src/
    ├── __init__.py
    ├── document_parser.py      # Parse PDF/DOCX with structure preservation
    ├── indexer.py              # FAISS vector index (embed & store)
    ├── summarizer.py           # Executive summary generation
    ├── clause_extractor.py     # Key clause extraction & translation
    ├── risk_analyzer.py        # Risk pattern detection & rating
    ├── conflict_detector.py    # Internal contradiction detection
    └── qa_chain.py             # RAG-based Q&A with source citations
```

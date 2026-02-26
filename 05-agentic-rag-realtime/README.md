# Agentic RAG with Real-Time Tools

A LangChain agent that decides **which tool to use** for every question — searching your internal documents, fetching live stock prices, current weather, Wikipedia articles, or the live web, depending on what the question needs.

---

## Agentic RAG vs Standard RAG

| Standard RAG | Agentic RAG |
|---|---|
| Always searches FAISS | Decides which tool(s) to use |
| One retrieval step | Multiple steps if needed |
| Only knows stored documents | Can fetch live data |
| Fast (single LLM call) | Slower (multi-step reasoning) |
| Deterministic path | Dynamic, question-driven path |

**When to use Agentic RAG:** When users ask mixed questions that combine internal knowledge with live data (e.g. "How does today's AAPL price compare to our internal valuation model?").

**When to use Standard RAG:** High-volume, low-latency workloads where every question is about static documents.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────┐
│  LLM Agent  │  ← reads tool descriptions to decide what to call
└──────┬──────┘
       │  ReAct Loop: Reason → Act → Observe → Repeat
       │
  ┌────┴────────────────────────────────────────┐
  │                  Tool Registry              │
  │                                             │
  │  ┌──────────────────┐  ┌─────────────────┐ │
  │  │ search_knowledge │  │  get_stock_data │ │
  │  │     _base        │  │  (yfinance)     │ │
  │  │  (FAISS index)   │  └─────────────────┘ │
  │  └──────────────────┘                       │
  │  ┌──────────────────┐  ┌─────────────────┐ │
  │  │   web_search     │  │   get_weather   │ │
  │  │  (Tavily API)    │  │  (OpenWeather)  │ │
  │  └──────────────────┘  └─────────────────┘ │
  │  ┌──────────────────┐                       │
  │  │ search_wikipedia │                       │
  │  │  (Wikipedia API) │                       │
  │  └──────────────────┘                       │
  └─────────────────────────────────────────────┘
       │
       ▼
  Final Answer + Sources
```

---

## How the Agent Decides Which Tool to Use

The agent's LLM reads every tool's `name` and `description` string before responding. Here is the decision process for a typical question:

**Question:** *"What is AAPL's current price and how does it compare to our internal forecast?"*

```
Step 1 — REASON:
  "This question needs current stock data AND internal documents.
   I should call get_stock_data first, then search_knowledge_base."

Step 2 — ACT: get_stock_data("AAPL")
Step 3 — OBSERVE: "Stock: AAPL | Price: $182.50 | ..."

Step 4 — REASON:
  "Now I have the live price. I need the internal forecast from the KB."

Step 5 — ACT: search_knowledge_base("AAPL valuation forecast")
Step 6 — OBSERVE: "Found in knowledge base: 1. Q3 forecast values AAPL at..."

Step 7 — REASON:
  "I have both pieces of information. I can now compose a full answer."

Step 8 — FINAL ANSWER (no more tool calls needed)
```

The key insight: **the tool description IS the routing logic**. A clear description like *"Use this for questions about internal policies"* routes the agent correctly without any if/else code.

---

## Setup

### 1. Clone and install

```bash
cd 05-agentic-rag-realtime
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your keys
```

### API Key Guide

| Service | Required? | Free Tier | Sign-up Link |
|---|---|---|---|
| **OpenAI** | ✅ Yes | Pay-per-use | [platform.openai.com](https://platform.openai.com) |
| **Tavily** (web search) | ❌ Optional | 1,000 searches/month | [tavily.com](https://tavily.com) |
| **OpenWeatherMap** | ❌ Optional | 60 calls/min | [openweathermap.org/api](https://openweathermap.org/api) |
| **yfinance** (finance) | ✅ Built-in | Unlimited* | No key needed |
| **Wikipedia** | ✅ Built-in | Unlimited | No key needed |

*yfinance scrapes Yahoo Finance; data may be delayed 15 minutes.

### 3. Add documents to the knowledge base (optional)

```bash
# Drop .pdf or .txt files here:
data/knowledge_base/
```

### 4. Run

```bash
# Single query
python main.py --query "What is AAPL's current stock price?"

# Interactive session
python main.py --interactive

# Without conversation memory
python main.py --interactive --no-memory

# Hide the reasoning trace
python main.py --query "Weather in Tokyo" --no-verbose
```

---

## Example Multi-Tool Queries

### Finance + RAG
**Query:** *"What is AAPL price and how does it compare to our internal valuation?"*

```
Agent calls: get_stock_data("AAPL") → search_knowledge_base("AAPL valuation")
```

### Weather + RAG
**Query:** *"What's the weather in London and should we proceed per our event guidelines?"*

```
Agent calls: get_weather("London") → search_knowledge_base("event guidelines weather policy")
```

### Web Search + RAG
**Query:** *"What are latest AI news stories relevant to our strategy?"*

```
Agent calls: web_search("latest AI news 2024") → search_knowledge_base("AI strategy")
```

### Wikipedia + Finance
**Query:** *"What does Wikipedia say about transformer models and how is NVDA performing?"*

```
Agent calls: search_wikipedia("transformer neural network") → get_stock_data("NVDA")
```

---

## Cost and Rate Limits

| Tool | Cost | Rate Limit |
|---|---|---|
| `search_knowledge_base` | Free (local FAISS) | Unlimited |
| `get_stock_data` | Free (yfinance) | ~2,000 req/hour* |
| `search_wikipedia` | Free | Unlimited |
| `web_search` | Free tier: 1,000/month | 1 req/sec |
| `get_weather` | Free tier: 60 calls/min | 1,000,000/month |
| OpenAI GPT-4 | ~$0.03/1K tokens | Depends on tier |
| OpenAI GPT-3.5 | ~$0.002/1K tokens | Depends on tier |

*Yahoo Finance has unofficial rate limits; excessive calls may trigger temporary blocks.

---

## How to Add a Custom Tool

Adding a new tool requires three steps: write the function, wrap it in a `Tool`, and register it.

### Step 1 — Write the function

Create `src/tools/my_tool.py`:

```python
from langchain.tools import Tool

def my_custom_function(input_str: str) -> str:
    # Your logic here — always string in, string out
    return "Result: ..."

def create_my_tool() -> Tool:
    return Tool(
        name="my_tool_name",
        func=my_custom_function,
        description=(
            "What this tool does and when to use it. "
            "Input: what to provide (be specific about format)."
        ),
    )
```

**Rules for a good tool:**
- Function signature: always `(input_str: str) -> str`
- Never raise exceptions — catch errors and return a message string
- Description must say WHAT the tool does, WHEN to use it, and WHAT input format it expects

### Step 2 — Register it in `tool_registry.py`

```python
from src.tools.my_tool import create_my_tool

def build_tool_registry(vector_store, config):
    tools = [...]  # existing tools
    tools.append(create_my_tool())
    return tools
```

### Step 3 — Test it

```bash
python main.py --query "A question that should trigger your new tool"
```

With `--verbose` (default) you'll see whether the agent picked your tool and what it returned.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `OPENAI_API_KEY not set` | Add key to `.env` file |
| Agent always uses `search_knowledge_base` | Knowledge base is empty — agent defaults to it. Add docs to `data/knowledge_base/` |
| `yfinance` returns `None` for price | Market may be closed; try a major ticker like AAPL |
| Web search returns mock message | Add `TAVILY_API_KEY` to `.env` |
| Weather returns mock data | Add `OPENWEATHERMAP_API_KEY` to `.env` |
| Agent uses wrong tool | Check tool descriptions in `src/tools/` — make them more specific |
| `FAISS` index error on reload | Delete `data/knowledge_base/.faiss_index/` and re-run |
| Agent loops > 8 times | Increase `max_iterations` in `src/agent.py` or simplify your query |
| `sentence-transformers` slow on first run | It downloads the model (~80 MB) once; subsequent runs are fast |

### Changing the LLM

```bash
# Use GPT-3.5 instead of GPT-4 (cheaper, slightly less accurate tool selection)
python main.py --model gpt-3.5-turbo --interactive
```

### Viewing the reasoning trace

The `--verbose` flag (on by default) prints every Thought → Action → Observation cycle. This is the best way to debug unexpected answers:

```
Thought: I need current stock data for AAPL.
Action: get_stock_data
Action Input: AAPL
Observation: Stock: AAPL | Price: $182.50 | ...
Thought: I now have the price. Let me check the knowledge base for the internal valuation.
...
```

---

## Project Structure

```
05-agentic-rag-realtime/
├── main.py                    # Entry point — pipeline + CLI
├── requirements.txt
├── .env.example               # Copy to .env and fill in keys
├── data/
│   └── knowledge_base/        # Drop .pdf and .txt files here
├── src/
│   ├── knowledge_indexer.py   # FAISS index builder (reused from Project 1)
│   ├── tool_registry.py       # Assembles all tools into a list
│   ├── agent.py               # LangChain agent with ReAct loop
│   ├── response_formatter.py  # Formats output with sources and trace
│   └── tools/
│       ├── rag_tool.py        # Wraps FAISS search as a Tool
│       ├── finance_tool.py    # yfinance stock data
│       ├── weather_tool.py    # OpenWeatherMap current weather
│       ├── web_search_tool.py # Tavily live web search
│       └── wiki_tool.py       # Wikipedia summaries
```

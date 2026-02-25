# 🤖 Project 5: Agentic RAG with Real-Time Data

An AI agent that **dynamically decides** which tools to use — combining a local knowledge base (RAG) with live data from the web, stock market, weather, and Wikipedia.

## Architecture

```
                         ┌─────────────────┐
                         │   User Query     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   LLM Agent      │
                         │  (Decision Hub)  │
                         │                  │
                         │  Reads tool      │
                         │  descriptions →  │
                         │  picks the best  │
                         │  tool for the    │
                         │  question        │
                         └───────┬─────────┘
                                 │
              ┌──────────┬───────┼───────┬──────────┐
              ▼          ▼       ▼       ▼          ▼
        ┌──────────┐ ┌───────┐ ┌─────┐ ┌───────┐ ┌──────┐
        │Knowledge │ │ Web   │ │Stock│ │Weather│ │Wiki  │
        │  Base    │ │Search │ │Data │ │  API  │ │pedia │
        │ (FAISS)  │ │(Serp) │ │(yf) │ │(OWM)  │ │      │
        └──────────┘ └───────┘ └─────┘ └───────┘ └──────┘
              │          │       │       │          │
              ▼          ▼       ▼       ▼          ▼
        ┌──────────────────────────────────────────────┐
        │         Agent Synthesizes Final Answer        │
        │     (combines results from multiple tools)    │
        └──────────────────────────────────────────────┘
```

## How the Agent Decides Which Tool to Use

The agent's LLM reads each tool's **name** and **description** at every turn. It then matches the user's query to the most relevant tool:

| User Query | Agent Reasoning | Tool Selected |
|---|---|---|
| "What does our policy say about..." | Internal documents → | `knowledge_base` |
| "What's Apple's stock price?" | Stock data → | `stock_market` |
| "Weather in Tokyo?" | Weather query → | `weather` |
| "Who was Alan Turing?" | Historical fact → | `wikipedia` |
| "Latest AI news today?" | Current events → | `web_search` |

The agent can also **chain multiple tools** in one query:
> "Compare Tesla's stock price with what Wikipedia says about the company"
> → Calls `stock_market` then `wikipedia`, combines both results.

## Setup Instructions

### 1. Install Dependencies

```bash
cd 05-agentic-rag-realtime
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys (see the API Key Guide below).

### 3. Add Documents (Optional)

Place PDF or text files in `data/knowledge_base/`:

```
data/knowledge_base/
  ├── company-policy.pdf
  ├── research-paper.pdf
  └── notes.txt
```

### 4. Run

```bash
python main.py
```

## API Key Setup Guide

| Service | Required? | Free Tier | Get a Key |
|---|---|---|---|
| **OpenAI** | Yes (or use Ollama) | $5 free credit | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Ollama** | Alternative to OpenAI | Completely free | [ollama.ai](https://ollama.ai) — run `ollama pull llama3` |
| **SerpAPI** | Optional | 100 searches/month | [serpapi.com](https://serpapi.com/) |
| **OpenWeatherMap** | Optional | 60 calls/min | [openweathermap.org/api](https://openweathermap.org/api) |
| **yfinance** | — | Free, no key needed | Included in requirements |
| **Wikipedia** | — | Free, no key needed | Included in requirements |

### Using Ollama (Free, Local)

If you don't want to use OpenAI, set these in your `.env`:

```env
USE_OLLAMA=true
OLLAMA_MODEL=llama3
```

Make sure Ollama is running: `ollama serve`

## Example Multi-Tool Queries

```
You: What's Apple's stock price and what does Wikipedia say about the company?
→ Agent calls: stock_market("AAPL") + wikipedia("Apple Inc")

You: What's the weather in London and any recent news about the UK?
→ Agent calls: weather("London") + web_search("UK news today")

You: Based on our documents, what's our vacation policy? Also, what's MSFT trading at?
→ Agent calls: knowledge_base("vacation policy") + stock_market("MSFT")
```

## Cost & Rate Limit Considerations

| Tool | Cost | Rate Limit | Notes |
|---|---|---|---|
| Knowledge Base | Free | None | Runs locally via FAISS |
| Wikipedia | Free | None | No API key needed |
| yfinance | Free | ~2,000/hour | May get throttled with heavy use |
| SerpAPI | Free tier: 100/mo | 100 searches/month | Pay-as-you-go after free tier |
| OpenWeatherMap | Free tier | 60 calls/min | 1M calls/month on free plan |
| OpenAI GPT-3.5 | ~$0.002/1K tokens | Varies by plan | ~$0.01 per agent query |

**Tip:** The agent prefers free tools (knowledge base, Wikipedia, yfinance) when possible. Web search is only used for current events.

## How to Add a Custom Tool

Adding a new tool takes just 3 steps:

### Step 1: Create the Tool File

Create `src/tools/my_custom_tool.py`:

```python
"""My Custom Tool — brief description of what it does."""

from langchain.tools import Tool


def create_my_custom_tool():
    """Create and return a LangChain Tool."""

    def _my_tool_function(query: str) -> str:
        """
        The actual logic of your tool.

        Args:
            query: The input string from the agent.

        Returns:
            A string with the tool's response.
        """
        # Your logic here — API calls, calculations, database queries, etc.
        result = f"Processed: {query}"
        return result

    tool = Tool(
        name="my_custom_tool",          # Unique name (no spaces)
        func=_my_tool_function,
        description=(
            "Describe WHEN the agent should use this tool. "
            "Be specific — the LLM reads this to make decisions. "
            "Input should be: describe expected input format."
        ),
    )

    return tool
```

### Step 2: Register It

In `src/tool_registry.py`, add:

```python
from src.tools.my_custom_tool import create_my_custom_tool

# Inside register_tools():
custom_tool = create_my_custom_tool()
tools.append(custom_tool)
print(f"  ✅ Registered: {custom_tool.name}")
```

### Step 3: Test It

Run the agent and ask a question that should trigger your tool. Watch the verbose output to confirm the agent selects it.

## Project Structure

```
05-agentic-rag-realtime/
├── main.py                    # Entry point — run this
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API keys
├── README.md                  # This file
├── data/
│   └── knowledge_base/        # Put your PDFs and text files here
├── src/
│   ├── __init__.py
│   ├── knowledge_indexer.py   # FAISS indexing & loading
│   ├── tool_registry.py       # Registers all tools for the agent
│   ├── agent.py               # Agent creation & execution
│   ├── response_formatter.py  # Clean output formatting
│   └── tools/
│       ├── __init__.py
│       ├── rag_tool.py        # Knowledge base retrieval
│       ├── web_search_tool.py # Live web search (SerpAPI)
│       ├── finance_tool.py    # Stock market data (yfinance)
│       ├── weather_tool.py    # Weather data (OpenWeatherMap)
│       └── wiki_tool.py       # Wikipedia lookups
└── faiss_index/               # Auto-generated FAISS index (gitignored)
```

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| "No valid OPENAI_API_KEY" | Set it in `.env` or switch to Ollama |
| Agent picks the wrong tool | Improve the tool's `description` text |
| Slow first run | Embedding model downloads ~90 MB on first use |
| "Ollama connection error" | Start Ollama with `ollama serve` |

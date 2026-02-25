"""
Finance Tool — Stock Market Data
=================================
Uses the `yfinance` library to fetch real-time stock information including
current price, 52-week high/low, and P/E ratio.

Ticker extraction:
  The agent's LLM is responsible for extracting the ticker symbol from the
  user's natural-language query.  For example:
    "What's Apple's stock price?" → the LLM sends "AAPL" to this tool.
  If the user says a company name instead of a ticker, the LLM usually
  knows the mapping (e.g., "Tesla" → "TSLA").

Handling missing tickers:
  - yfinance returns empty data for invalid tickers — we check for that.
  - We also catch network errors so the agent can report them gracefully.
"""

import yfinance as yf
from langchain.tools import Tool


def create_finance_tool():
    """
    Create a LangChain Tool that fetches stock market data via yfinance.

    Returns:
        A LangChain Tool the agent can invoke by name.
    """

    def _get_stock_info(query: str) -> str:
        """
        Fetch stock data for a given ticker symbol.

        Args:
            query: A stock ticker symbol like "AAPL", "MSFT", "TSLA".
                   The agent should extract the ticker from the user's question.
        """
        # Clean the input — users (or the LLM) might include extra spaces
        ticker_symbol = query.strip().upper()

        if not ticker_symbol:
            return "Please provide a valid stock ticker symbol (e.g., AAPL, MSFT)."

        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            # --- Handle case where ticker isn't found ---
            # yfinance doesn't raise an error for bad tickers; it returns
            # a dict with minimal or default values.  We check for a
            # missing 'currentPrice' (or 'regularMarketPrice') as a signal.
            current_price = info.get("currentPrice") or info.get(
                "regularMarketPrice"
            )
            if current_price is None:
                return (
                    f"Could not find data for ticker '{ticker_symbol}'. "
                    f"Make sure it's a valid stock symbol (e.g., AAPL for Apple)."
                )

            # Extract key metrics
            company_name = info.get("shortName", ticker_symbol)
            currency = info.get("currency", "USD")
            week_high_52 = info.get("fiftyTwoWeekHigh", "N/A")
            week_low_52 = info.get("fiftyTwoWeekLow", "N/A")
            pe_ratio = info.get("trailingPE", "N/A")
            market_cap = info.get("marketCap", "N/A")

            # Format market cap for readability
            if isinstance(market_cap, (int, float)):
                if market_cap >= 1e12:
                    market_cap_str = f"${market_cap / 1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap_str = f"${market_cap / 1e9:.2f}B"
                elif market_cap >= 1e6:
                    market_cap_str = f"${market_cap / 1e6:.2f}M"
                else:
                    market_cap_str = f"${market_cap:,.0f}"
            else:
                market_cap_str = str(market_cap)

            # Format P/E ratio
            pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, float) else str(pe_ratio)

            return (
                f"📈 {company_name} ({ticker_symbol})\n"
                f"  Current Price : {currency} {current_price:,.2f}\n"
                f"  52-Week High  : {currency} {week_high_52}\n"
                f"  52-Week Low   : {currency} {week_low_52}\n"
                f"  P/E Ratio     : {pe_str}\n"
                f"  Market Cap    : {market_cap_str}"
            )

        except Exception as e:
            # Network errors, API issues, etc. — let the agent know.
            return (
                f"Error fetching stock data for '{ticker_symbol}': {str(e)}. "
                f"Check your internet connection or try a different ticker."
            )

    tool = Tool(
        name="stock_market",
        func=_get_stock_info,
        description=(
            "Use this to get current stock market data for a company. "
            "Input should be a stock ticker symbol (e.g., AAPL, MSFT, TSLA, GOOGL). "
            "Returns current price, 52-week high/low, P/E ratio, and market cap. "
            "If the user mentions a company name, convert it to a ticker first."
        ),
    )

    return tool

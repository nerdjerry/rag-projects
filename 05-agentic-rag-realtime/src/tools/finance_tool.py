"""
src/tools/finance_tool.py

Fetches live stock and financial data via yfinance (Yahoo Finance).

This is a great example tool because it has:
    • A concrete, unambiguous input format: the ticker symbol (AAPL, MSFT, …)
    • A concrete, structured output: price, range, P/E, market cap
    • No API key required — yfinance scrapes Yahoo Finance directly

LIMITATIONS:
    • Data may be delayed up to 15 minutes (Yahoo Finance's standard delay).
    • Market cap and P/E are sourced from Yahoo Finance's "info" dict, which
      can occasionally be None for smaller or recently-listed companies.
    • For company-name inputs (e.g. "Apple") we do a best-effort ticker lookup
      using yfinance's search; this may not always resolve correctly.
"""

from langchain.tools import Tool


# Common company-name → ticker fallback mapping for the most-searched names.
# yfinance doesn't have a built-in name→ticker resolver so we keep a small
# local table for robustness. Users who pass valid tickers bypass this table.
_NAME_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "berkshire": "BRK-B",
    "visa": "V",
    "jpmorgan": "JPM",
    "walmart": "WMT",
}


def _resolve_ticker(input_str: str) -> str:
    """
    Best-effort conversion of a user's input to a valid ticker symbol.

    Priority:
        1. If the input looks like a ticker (short, uppercase, no spaces) use it.
        2. Check the local name→ticker mapping.
        3. Fall back to the input as-is (yfinance will fail if it's wrong).
    """
    stripped = input_str.strip()

    # Heuristic: tickers are 1–5 uppercase letters (optionally with a dot/dash)
    if len(stripped) <= 6 and stripped.replace("-", "").replace(".", "").isalpha():
        return stripped.upper()

    # Check local lookup table (case-insensitive)
    lower = stripped.lower()
    for name, ticker in _NAME_TO_TICKER.items():
        if name in lower:
            return ticker

    # Last resort: return input uppercased and hope it's a valid ticker
    return stripped.upper()


def create_finance_tool() -> Tool:
    """
    Build a LangChain Tool that returns live stock data from Yahoo Finance.

    No API key is required. yfinance handles all HTTP communication internally.

    Returns:
        A configured LangChain Tool for stock data lookups.
    """

    def get_stock_data(input_str: str) -> str:
        """
        Fetch key financial metrics for a given ticker or company name.

        The agent passes the ticker or company name as a plain string.
        We return a single formatted line so the agent can include it verbatim
        in its response without further parsing.
        """
        try:
            import yfinance as yf  # noqa: PLC0415

            ticker_symbol = _resolve_ticker(input_str)
            ticker = yf.Ticker(ticker_symbol)

            # fast_info is lighter-weight than the full .info dict and avoids
            # some rate-limiting issues, but has fewer fields.
            fast = ticker.fast_info
            info = ticker.info  # full metadata dict — may be slow on first call

            # Safely extract values; Yahoo Finance sometimes returns None.
            price = fast.last_price
            if price is None:
                return (
                    f"Could not find stock data for '{input_str}'. "
                    "Please provide a valid ticker symbol."
                )

            high_52w = fast.year_high
            low_52w = fast.year_low
            market_cap = fast.market_cap
            pe_ratio = info.get("trailingPE")

            # --- Format market cap as a human-readable string ---
            def _fmt_cap(cap) -> str:
                if cap is None:
                    return "N/A"
                if cap >= 1e12:
                    return f"${cap / 1e12:.2f}T"
                if cap >= 1e9:
                    return f"${cap / 1e9:.2f}B"
                if cap >= 1e6:
                    return f"${cap / 1e6:.2f}M"
                return f"${cap:,.0f}"

            pe_str = f"{pe_ratio:.1f}" if pe_ratio else "N/A"
            high_str = f"${high_52w:.2f}" if high_52w else "N/A"
            low_str = f"${low_52w:.2f}" if low_52w else "N/A"

            return (
                f"Stock: {ticker_symbol} | "
                f"Price: ${price:.2f} | "
                f"52W High: {high_str} | "
                f"52W Low: {low_str} | "
                f"P/E: {pe_str} | "
                f"Market Cap: {_fmt_cap(market_cap)}"
            )

        except Exception as exc:
            # Catch-all so a yfinance network error doesn't crash the agent loop.
            return (
                f"Could not find stock data for '{input_str}'. "
                f"Error: {exc}. "
                "Please provide a valid ticker symbol (e.g., AAPL, MSFT, GOOGL)."
            )

    return Tool(
        name="get_stock_data",
        func=get_stock_data,
        description=(
            "Get current stock/financial data for a publicly traded company. "
            "Input: a stock ticker symbol (e.g., AAPL, MSFT, GOOGL) or company name. "
            "Returns current price, 52-week range, P/E ratio, and market cap."
        ),
    )

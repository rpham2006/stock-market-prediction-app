r"""
Entry point — uses the stockkit package.

Note what this file does NOT contain: no URLs, no JSON parsing, no
HTTP. That's all behind the package's interface. This file only
expresses intent.

Run:  .venv\Scripts\python.exe main.py
"""

from stockkit import (
    MarketDataClient,
    Quote,
    SymbolNotFoundError,
    YahooFinanceClient,
)


def show_quote(client: MarketDataClient, symbol: str) -> Quote | None:
    """Fetch and print one quote.

    Note the parameter type is the INTERFACE, not the concrete class.
    Swap in a Finnhub client later and this function is unchanged.
    """
    try:
        quote = client.get_quote(symbol)
    except SymbolNotFoundError as exc:
        print(f"  [skipped] {exc}")
        return None

    print(f"  {quote}")
    print(f"      day range position: {quote.position_in_day_range:.0%} "
          f"(0% = at the low, 100% = at the high)")
    print(f"      100 shares would be worth ${quote.value_of(100):,.2f}")
    return quote


def main() -> None:
    # Depend on the interface; construct the implementation once.
    client: MarketDataClient = YahooFinanceClient(timeout_seconds=20)

    print("=" * 64)
    print("  QUOTES")
    print("=" * 64)

    for symbol in ["AAPL", "MSFT", "NVDA", "NOTAREALTICKER"]:
        show_quote(client, symbol)

    print()
    print("=" * 64)
    print("  AAPL — LAST 10 TRADING DAYS")
    print("=" * 64)
    print()

    bars = client.get_history("AAPL", days=14)

    for bar in bars[-10:]:
        print(f"  {bar}")          # calls PriceBar.__str__

    # ---- work with the objects, not raw dictionaries ----
    up_days = [bar for bar in bars if bar.is_up_day]
    best = max(bars, key=lambda bar: bar.percent_change)
    widest = max(bars, key=lambda bar: bar.trading_range)

    print()
    print(f"  Up days:        {len(up_days)} of {len(bars)}")
    print(f"  Best day:       {best.day} at {best.percent_change:+.2f}%")
    print(f"  Widest range:   {widest.day} spanning ${widest.trading_range:.2f}")
    print()
    print(f"  {client!r}")        # calls YahooFinanceClient.__repr__


if __name__ == "__main__":
    main()

r"""
LESSON 2 — Get a real stock price from the internet.

This uses ONLY the Python from Lesson 1: variables, f-strings, lists,
dictionaries, loops, and functions. Nothing new except one import.

Run:  .venv\Scripts\python.exe learn\02_get_a_stock.py
"""

# An import pulls in code someone else wrote so you can use it.
# yfinance knows how to talk to Yahoo Finance's servers.
import yfinance as yf          # "as yf" = give it a short nickname


# ---------------------------------------------------------------
# STEP 1 — Ask for one stock
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1 — Getting one stock")
print("=" * 60)

# yf.Ticker() doesn't download anything yet. It just creates an object
# that KNOWS HOW to go get Apple's data when you ask it to.
apple = yf.Ticker("AAPL")

# .info goes out to the internet and comes back with a DICTIONARY --
# the exact thing you learned in Lesson 1, section 4.
info = apple.info

print(f"\nWhat came back is a {type(info)} with {len(info)} keys.")
print("Same key -> value structure you already know.\n")

# Look things up by key, exactly like Lesson 1.
print(f"  Company:      {info['longName']}")
print(f"  Symbol:       {info['symbol']}")
print(f"  Sector:       {info['sector']}")
print(f"  Current price: ${info['currentPrice']}")


# ---------------------------------------------------------------
# STEP 2 — Get the last 5 days of prices
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 — The last 5 days")
print("=" * 60)

# .history() downloads past prices. period="5d" means "last 5 days".
history = apple.history(period="5d")

# This comes back as a TABLE (a "DataFrame"), not a dictionary.
# Think of it as a spreadsheet: rows are dates, columns are values.
print()
print(history)


# ---------------------------------------------------------------
# STEP 3 — Pull one column out and loop over it
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 — Just the closing prices")
print("=" * 60)
print()

# history["Close"] grabs a single column, like a dict lookup.
closing_prices = history["Close"]

# Looping over it gives you each date and its price.
for date, price in closing_prices.items():
    print(f"  {date:%Y-%m-%d}   ${price:.2f}")


# ---------------------------------------------------------------
# STEP 4 — Do something with the numbers
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 — Did it go up?")
print("=" * 60)

# .iloc[0] is the first row, .iloc[-1] is the last row.
first_price = closing_prices.iloc[0]
last_price = closing_prices.iloc[-1]

change = last_price - first_price
percent_change = (change / first_price) * 100

print(f"\n  5 days ago:  ${first_price:.2f}")
print(f"  Today:       ${last_price:.2f}")
print(f"  Change:      ${change:+.2f}  ({percent_change:+.2f}%)")

# An if/else from Lesson 1, section 6.
if change > 0:
    print(f"\n  AAPL went UP over the last 5 days.")
else:
    print(f"\n  AAPL went DOWN over the last 5 days.")


# ---------------------------------------------------------------
# STEP 5 — Wrap it in a function and do several stocks
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 — Several stocks at once")
print("=" * 60)
print()

def get_5day_change(symbol: str) -> float:
    """Return the percent change for one stock over the last 5 days."""
    stock = yf.Ticker(symbol)
    prices = stock.history(period="5d")["Close"]
    first = prices.iloc[0]
    last = prices.iloc[-1]
    return ((last - first) / first) * 100


# A list from Lesson 1, section 3.
watchlist = ["AAPL", "MSFT", "NVDA", "GOOGL"]

# A loop from Lesson 1, section 5.
for symbol in watchlist:
    pct = get_5day_change(symbol)
    arrow = "UP  " if pct > 0 else "DOWN"
    print(f"  {symbol:<6} {arrow}  {pct:+6.2f}%")

print("\nThat's it. You just pulled live market data off the internet.")

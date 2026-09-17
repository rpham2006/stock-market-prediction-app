r"""
LESSON 1 — Python basics, using stock examples.

You already know programming. This is just Python's spelling of things
you've seen before. Run it and read the output next to the code.

Run:  .venv\Scripts\python.exe learn\01_python_basics.py
"""

# ---------------------------------------------------------------
# 1. VARIABLES — no type declarations, no semicolons
# ---------------------------------------------------------------
# Java:   String ticker = "AAPL";
# Python: ticker = "AAPL"

ticker = "AAPL"
price = 227.50          # a decimal number is a "float"
shares = 10             # a whole number is an "int"
is_open = True          # boolean — note the capital T

print("--- 1. Variables ---")
print(ticker, price, shares, is_open)


# ---------------------------------------------------------------
# 2. PRINTING WITH VALUES INSIDE — the f-string
# ---------------------------------------------------------------
# Put an f before the quote, then {} around any variable.
# This is the single most useful piece of Python syntax.

total = price * shares
print(f"\n--- 2. f-strings ---")
print(f"{shares} shares of {ticker} at ${price} = ${total}")

# You can format numbers inside the braces too:
print(f"Formatted to 2 decimals: ${total:.2f}")


# ---------------------------------------------------------------
# 3. LISTS — an ordered collection (like an array)
# ---------------------------------------------------------------
print("\n--- 3. Lists ---")

tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]

print(f"The whole list:  {tickers}")
print(f"First item:      {tickers[0]}")      # counting starts at 0
print(f"Last item:       {tickers[-1]}")     # -1 means "from the end"
print(f"How many:        {len(tickers)}")

tickers.append("TSLA")                       # add to the end
print(f"After append:    {tickers}")


# ---------------------------------------------------------------
# 4. DICTIONARIES — labeled values (key -> value)
# ---------------------------------------------------------------
# Like a HashMap or a JS object. This is how API data arrives.
print("\n--- 4. Dictionaries ---")

stock = {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "price": 227.50,
}

print(f"Whole dict:  {stock}")
print(f"Just price:  {stock['price']}")      # look up by key
stock["sector"] = "Technology"               # add a new key
print(f"After adding sector: {stock}")


# ---------------------------------------------------------------
# 5. LOOPS — indentation defines the block, not { }
# ---------------------------------------------------------------
# This is the big visual difference from Java/C/JS.
# The colon starts the block; the indentation IS the block.
print("\n--- 5. Loops ---")

for t in tickers:
    print(f"  Checking {t}...")


# ---------------------------------------------------------------
# 6. IF / ELSE
# ---------------------------------------------------------------
print("\n--- 6. If / else ---")

for t in tickers:
    if t == "AAPL":
        print(f"  {t} — this is the one we care about")
    elif t == "TSLA":
        print(f"  {t} — too volatile, skipping")
    else:
        print(f"  {t} — adding to watchlist")


# ---------------------------------------------------------------
# 7. FUNCTIONS
# ---------------------------------------------------------------
# def instead of "public static". The -> float is optional; it just
# documents what comes back.
print("\n--- 7. Functions ---")

def position_value(price: float, shares: int) -> float:
    """Return the dollar value of a position."""
    return price * shares


print(f"  100 shares at $227.50 = ${position_value(227.50, 100):,.2f}")
print(f"  50 shares at $415.20  = ${position_value(415.20, 50):,.2f}")


# ---------------------------------------------------------------
# 8. PUTTING IT TOGETHER
# ---------------------------------------------------------------
# A list of dictionaries — this is exactly the shape data comes back
# in from a real API.
print("\n--- 8. All together ---")

portfolio = [
    {"symbol": "AAPL", "price": 227.50, "shares": 10},
    {"symbol": "MSFT", "price": 415.20, "shares": 5},
    {"symbol": "NVDA", "price": 178.90, "shares": 20},
]

grand_total = 0

for holding in portfolio:
    value = position_value(holding["price"], holding["shares"])
    grand_total = grand_total + value
    print(f"  {holding['symbol']:<6} {holding['shares']:>3} shares  =  ${value:>10,.2f}")

print(f"  {'TOTAL':<6} {'':>3}         =  ${grand_total:>10,.2f}")

print("\nThat's every piece of Python you need to fetch a stock price.")

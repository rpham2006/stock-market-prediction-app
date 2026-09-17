r"""
LESSON 3 — How the code actually gets data from the internet.

In Lesson 2, yfinance did the work and it looked like magic.
It isn't. This file does the SAME THING by hand, so you can see
every step.

Nothing new is installed here. urllib and json ship with Python.

Run:  .venv\Scripts\python.exe learn\03_how_it_gets_data.py
"""

import urllib.request   # makes internet requests  (built into Python)
import json             # reads JSON text            (built into Python)


# ===============================================================
# THE MENTAL MODEL
# ===============================================================
#
#   YOUR CODE                          YAHOO'S SERVER
#   (the "client")                     (the "server")
#        |                                    |
#        |  1. REQUEST                        |
#        |  "GET me /chart/AAPL"              |
#        | ---------------------------------> |
#        |                                    |  2. looks it up
#        |  3. RESPONSE                       |
#        |  "200 OK" + a big block of text    |
#        | <--------------------------------- |
#        |                                    |
#   4. you turn that text into a dictionary
#
# That's it. You send a URL, you get text back. Everything else
# on the internet is a variation of these four steps.
# ===============================================================


# ---------------------------------------------------------------
# STEP 1 — Build the URL
# ---------------------------------------------------------------
print("=" * 62)
print("STEP 1 — The URL (this is the whole 'request')")
print("=" * 62)

symbol = "AAPL"
url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d"

print(f"""
  {url}

  Read it in pieces:
    https://              the protocol (s = encrypted)
    query1.finance...     WHICH computer to talk to
    /v8/finance/chart/    WHICH function on that computer
    AAPL                  the stock we want
    ?range=5d&interval=1d parameters, after the '?', joined by '&'

  Paste that URL into your browser -- you'll see the same text
  this script is about to receive. A browser is just a program
  that does exactly what we're doing here.
""")


# ---------------------------------------------------------------
# STEP 2 — Send the request
# ---------------------------------------------------------------
print("=" * 62)
print("STEP 2 — Sending it")
print("=" * 62)

# Headers are extra notes attached to the request. User-Agent says
# who is asking. Many servers reject requests that don't identify
# themselves, so we claim to be a normal browser.
request = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0"},
)

# urlopen() is the actual moment your computer talks to Yahoo's.
# Everything before this line was just preparation.
response = urllib.request.urlopen(request, timeout=20)

print(f"""
  Status code: {response.status}

  200 means OK. You've seen the failures before:
    404  not found       (bad ticker, bad URL)
    401  unauthorized    (missing/invalid API key)
    429  too many requests (you hit the rate limit)
    500  server broke    (their fault, not yours)
""")


# ---------------------------------------------------------------
# STEP 3 — Read the response — it's just TEXT
# ---------------------------------------------------------------
print("=" * 62)
print("STEP 3 — What came back")
print("=" * 62)

raw_bytes = response.read()          # raw 1s and 0s
raw_text = raw_bytes.decode("utf-8") # turn them into readable characters

print(f"""
  Received {len(raw_text):,} characters of text.

  Here are the first 300:

{raw_text[:300]}...
""")

print("""  That's JSON -- text arranged with {curly braces} and "quotes".
  It is a dictionary that has been flattened into a string so it
  can travel over a wire.
""")


# ---------------------------------------------------------------
# STEP 4 — Turn the text into a Python dictionary
# ---------------------------------------------------------------
print("=" * 62)
print("STEP 4 — Text -> dictionary")
print("=" * 62)

data = json.loads(raw_text)   # "loads" = load-from-string

print(f"""
  json.loads() converted that text into a real Python {type(data).__name__}.

  Now it's the same thing from Lesson 1 -- look things up by key.
  Top-level keys: {list(data.keys())}
""")


# ---------------------------------------------------------------
# STEP 5 — Dig down to the numbers
# ---------------------------------------------------------------
print("=" * 62)
print("STEP 5 — Finding the price inside")
print("=" * 62)

# JSON nests dictionaries inside dictionaries inside lists.
# Each step below goes one level deeper.
chart = data["chart"]            # a dictionary
result = chart["result"]         # a LIST of results
first = result[0]                # the first (only) result -- a dictionary
meta = first["meta"]             # a dictionary of summary info

print(f"""
  data["chart"]["result"][0]["meta"] contains:

    Symbol:        {meta['symbol']}
    Exchange:      {meta['fullExchangeName']}
    Currency:      {meta['currency']}
    Current price: ${meta['regularMarketPrice']:.2f}
    Day high:      ${meta['regularMarketDayHigh']:.2f}
    Day low:       ${meta['regularMarketDayLow']:.2f}
""")

# The actual daily closing prices are buried a bit deeper.
closes = first["indicators"]["quote"][0]["close"]
timestamps = first["timestamp"]

print("  The last 5 daily closes:\n")
for ts, close in zip(timestamps, closes):
    if close is not None:
        print(f"    {close:.2f}")


# ---------------------------------------------------------------
# STEP 6 — So what is yfinance?
# ---------------------------------------------------------------
print()
print("=" * 62)
print("STEP 6 — What yfinance actually is")
print("=" * 62)
print("""
  Everything above -- build URL, add headers, send request, read
  text, parse JSON, dig through nested keys -- is what yfinance
  does for you when you write:

      apple = yf.Ticker("AAPL")
      apple.history(period="5d")

  It is a WRAPPER. Someone wrote those steps once, handled the
  errors and edge cases, and published it so nobody has to redo it.

  That's what a "library" is. Not magic -- just code someone else
  already wrote.

  Same story for every API you'll ever use:
      the Anthropic library wraps requests to Claude
      a database library wraps requests to Postgres

  Underneath, all of them are doing STEP 1 through STEP 5.
""")

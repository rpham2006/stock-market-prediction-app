"""
The web application package.

Layering, outermost to innermost:

    api.py / web.py   HTTP — routing, status codes, templates
    service.py        orchestration — refresh policy, assembling payloads
    repository.py     persistence — ORM rows in, value objects out
    predictor.py      the model
    stockkit/         market data (a separate package; knows nothing of this one)

Each layer may call inward, never outward. That is what makes the model
testable without a database and the database testable without a network.
"""

__version__ = "1.0.0"

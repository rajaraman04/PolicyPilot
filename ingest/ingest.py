"""Deprecated entry point — kept so `python -m ingest.ingest` still works.

The real pipeline lives in ingest/build.py. Prefer `python -m ingest.build`.
"""

from ingest.build import build, main

__all__ = ["build", "main"]


if __name__ == "__main__":
    main()

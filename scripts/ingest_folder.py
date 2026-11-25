"""Index every PDF in a folder.

Usage:  python -m scripts.ingest_folder data/raw
"""
import logging
import sys

from app.pipeline import RuleBook

logging.basicConfig(level=logging.INFO, format="%(message)s")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    rb = RuleBook()
    for stats in rb.ingest_folder(folder):
        print(stats)
    print(f"Total chunks in index: {len(rb.chunks)}")

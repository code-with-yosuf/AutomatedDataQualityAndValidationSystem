"""Kept for backwards compatibility; use pipeline.py."""
import sys

from pipeline import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Entrypoint for the AI Harness Arena JSON-lines protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.core import run


if __name__ == "__main__":
    run()

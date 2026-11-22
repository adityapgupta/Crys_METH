"""Frozen reference-selection evaluation; see README.md and evaluation.ipynb."""

from pathlib import Path
import sys

_UTILITIES = Path(__file__).resolve().parents[1]
if _UTILITIES.is_dir() and str(_UTILITIES) not in sys.path:
    sys.path.insert(0, str(_UTILITIES))

"""Resolve appendix imports to the baseline package bundled with Crys_METH."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_BASE_PATH = ROOT / "gnn"
for path in (ROOT, REFERENCE_BASE_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

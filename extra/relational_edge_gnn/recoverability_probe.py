"""Compatibility entry point for reference_baselines.evaluation.embedding_recoverability_probe."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # Resolves the repository's shared baseline package.

if __name__ == "__main__":
    import runpy
    runpy.run_module("reference_baselines.evaluation.embedding_recoverability_probe", run_name="__main__")
else:
    import importlib
    sys.modules[__name__] = importlib.import_module("reference_baselines.evaluation.embedding_recoverability_probe")

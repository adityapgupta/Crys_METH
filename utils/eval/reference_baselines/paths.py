"""Repository paths shared by every experiment, independent of the working directory."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
# This utility package is vendored beneath Crys_METH/utils/eval.
ROOT = PACKAGE_DIR.parents[2]
DATA_DIR = ROOT / "data/eval" / "inputs"
RESULTS_DIR = ROOT / "data/eval" / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "nonparametric"
CONFIG_DIR = PACKAGE_DIR / "configs"
DOCS_DIR = PACKAGE_DIR / "docs"

SOURCE_PACKAGES = (
    "core",
    "datasets",
    "graphs",
    "baselines",
    "structure",
    "evaluation",
    "models",
    "training",
    "checks",
    "reporting",
    "embedding_structure",
)

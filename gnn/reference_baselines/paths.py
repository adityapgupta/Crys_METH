"""Repository paths shared by every experiment, independent of the working directory."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
# The implementation lives under Crys_METH/gnn/reference_baselines.
ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = ROOT / "data/eval" / "inputs"
RESULTS_DIR = ROOT / "outputs/gnn"
BASELINE_RESULTS_DIR = RESULTS_DIR / "nonparametric"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
COMPARISON_DIR = RESULTS_DIR / "reference_set_comparison"
COMPARISON_DATA_DIR = ROOT / "data/eval/reference_set_comparison"
CONFIG_DIR = PACKAGE_DIR / "configs"
DOCS_DIR = RESULTS_DIR / "reports"

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

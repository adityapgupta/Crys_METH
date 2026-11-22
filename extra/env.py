"""Make the shared ``reference_baselines`` base package importable from here.

The appendix holds only each experiment's own code. Everything shared
(the base dataset, GNN, trainer, evaluator and precomputed inputs) lives in the
``reference_baselines`` package maintained alongside this folder. Set
``REFERENCE_BASE_PATH`` in a local ``.env`` to the directory that *contains* that
package; this module reads it and puts it on ``sys.path`` so that
``import reference_baselines.*`` resolves. Import this module before any
``reference_baselines`` import.
"""

import os
import sys
from pathlib import Path

APPENDIX_DIR = Path(__file__).resolve().parent


def _load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def reference_base_path():
    _load_env_file(APPENDIX_DIR / ".env")
    raw = os.environ.get("REFERENCE_BASE_PATH")
    if not raw:
        raise RuntimeError(
            "REFERENCE_BASE_PATH is not set. Copy .env.example to .env and point it "
            "at the directory that contains the reference_baselines package."
        )
    base = Path(raw).expanduser().resolve()
    if not (base / "reference_baselines").is_dir():
        raise RuntimeError(f"No reference_baselines package under REFERENCE_BASE_PATH={base}")
    return base


REFERENCE_BASE_PATH = reference_base_path()
if str(REFERENCE_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(REFERENCE_BASE_PATH))

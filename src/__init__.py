"""
src/__init__.py
---------------
Makes `src` a proper Python package.

Exposes the key public API of the project so any notebook or script
can import cleanly with a single line, e.g.:

    from src import PTBXLDataLoader
    from src import LEAD_NAMES

As you populate model.py, preprocessor.py, train.py, evaluate.py —
add their imports here the same way.
"""

# ── Data Loading ─────────────────────────────────────────────────────────────
from src.data_loader import (
    PTBXLDataLoader,   # Main loader class
    check_internet,    # Internet check helper
    LEAD_NAMES,        # ['I','II','III','aVR','aVL','aVF','V1',...,'V6']
    PHYSIONET_DB,      # Remote wfdb path constant
    LOCAL_DB_PATH,     # Local fallback path constant
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
from src.preprocessor import ECGPreprocessor

# ── Model ─────────────────────────────────────────────────────────────────────
# from src.model import ECGModel                 # ← uncomment when ready

# ── Training ──────────────────────────────────────────────────────────────────
# from src.train import train_model              # ← uncomment when ready

# ── Evaluation ────────────────────────────────────────────────────────────────
# from src.evaluate import evaluate_model        # ← uncomment when ready

__all__ = [
    # data_loader
    "PTBXLDataLoader",
    "check_internet",
    "LEAD_NAMES",
    "PHYSIONET_DB",
    "LOCAL_DB_PATH",
    # preprocessor
    "ECGPreprocessor",
]

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
from src.model import LSTMForecast, CNNLSTMForecast, TransformerForecast

# ── Training ──────────────────────────────────────────────────────────────────
from src.train import ECGForecastDataset, get_dataloaders, train_model

# ── Evaluation ────────────────────────────────────────────────────────────────
from src.evaluate import calculate_forecasting_metrics, print_metrics_table

__all__ = [
    # data_loader
    "PTBXLDataLoader",
    "check_internet",
    "LEAD_NAMES",
    "PHYSIONET_DB",
    "LOCAL_DB_PATH",
    # preprocessor
    "ECGPreprocessor",
    # model
    "LSTMForecast",
    "CNNLSTMForecast",
    "TransformerForecast",
    # train
    "ECGForecastDataset",
    "get_dataloaders",
    "train_model",
    # evaluate
    "calculate_forecasting_metrics",
    "print_metrics_table",
]

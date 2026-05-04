"""
src/data_loader.py
------------------
PTB-XL data loading utility used by downstream notebooks and training scripts.

Priority:
  1. Online  — streams directly from PhysioNet (no local files needed)
  2. Offline — falls back to data/physionet.org/files/ptb-xl/1.0.3/ if present
"""

import os
import ast
import urllib.request

import wfdb
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer


# ── Constants ────────────────────────────────────────────────────────────────
PHYSIONET_DB  = 'ptb-xl/1.0.3'
LOCAL_DB_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data',
    'physionet.org', 'files', 'ptb-xl', '1.0.3'
)

ONLINE_CSV = 'https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
ONLINE_SCP = 'https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv'

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


# ── Helpers ──────────────────────────────────────────────────────────────────
def check_internet(timeout: int = 5) -> bool:
    """Returns True if PhysioNet is reachable."""
    try:
        urllib.request.urlopen('https://physionet.org', timeout=timeout)
        return True
    except Exception:
        return False


def _split_path(raw_path: str):
    """Split a ptbxl filename_lr value into (folder, rec_name)."""
    parts    = raw_path.split('/')
    folder   = '/'.join(parts[:-1])
    rec_name = parts[-1]
    return folder, rec_name


# ── Core Loader Class ────────────────────────────────────────────────────────
class PTBXLDataLoader:
    """
    Loads PTB-XL metadata and waveforms.

    Parameters
    ----------
    sampling_rate : int
        100 (default) or 500 Hz.

    Usage
    -----
    loader = PTBXLDataLoader(sampling_rate=100)
    df     = loader.load_metadata()
    train, val, test = loader.get_data_splits()
    X, y   = loader.load_waveforms(train.head(200))
    """

    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        self.mode          = 'online' if check_internet() else 'offline'
        self.metadata      = None
        self.scp_statements = None
        self.mlb           = MultiLabelBinarizer()
        self.classes_      = None

        print(f"[PTBXLDataLoader] mode={self.mode} | fs={self.sampling_rate} Hz")
        if self.mode == 'offline':
            local_csv = os.path.join(LOCAL_DB_PATH, 'ptbxl_database.csv')
            if not os.path.exists(local_csv):
                raise FileNotFoundError(
                    "No internet AND no local data found.\n"
                    f"Expected: {local_csv}\n"
                    "Connect to the internet, or place the PTB-XL dataset in data/physionet.org/."
                )

    # ── Metadata ─────────────────────────────────────────────────────────────
    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata CSV and SCP statements.
        Returns a DataFrame with scp_codes parsed and superclass labels applied.
        Multi-hot encoded label columns (NORM, MI, STTC, CD, HYP) are appended.
        """
        if self.mode == 'online':
            df     = pd.read_csv(ONLINE_CSV, index_col='ecg_id')
            scp_df = pd.read_csv(ONLINE_SCP,  index_col=0)
        else:
            df     = pd.read_csv(os.path.join(LOCAL_DB_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
            scp_df = pd.read_csv(os.path.join(LOCAL_DB_PATH, 'scp_statements.csv'),  index_col=0)

        df.scp_codes         = df.scp_codes.apply(ast.literal_eval)
        self.scp_statements  = scp_df

        # Map to diagnostic superclasses
        diag_scp = scp_df[scp_df.diagnostic == 1]

        def get_superclass(scp_dict):
            return [diag_scp.loc[k].diagnostic_class
                    for k in scp_dict if k in diag_scp.index]

        df['superclass'] = df.scp_codes.apply(get_superclass)

        # Multi-hot encode
        multi_hot    = self.mlb.fit_transform(df['superclass'])
        self.classes_ = self.mlb.classes_
        mlb_df       = pd.DataFrame(multi_hot, columns=self.classes_, index=df.index)
        self.metadata = pd.concat([df, mlb_df], axis=1)

        print(f"Loaded {len(self.metadata):,} records | classes: {list(self.classes_)}")
        return self.metadata

    # ── Splits ───────────────────────────────────────────────────────────────
    def get_data_splits(self):
        """
        Returns (train, val, test) DataFrames using the official PTB-XL fold split:
            Folds 1-8 → Train | Fold 9 → Validation | Fold 10 → Test
        """
        if self.metadata is None:
            self.load_metadata()
        train = self.metadata[self.metadata.strat_fold <= 8]
        val   = self.metadata[self.metadata.strat_fold == 9]
        test  = self.metadata[self.metadata.strat_fold == 10]
        print(f"Split → train={len(train):,} | val={len(val):,} | test={len(test):,}")
        return train, val, test

    # ── Waveforms ─────────────────────────────────────────────────────────────
    def load_waveforms(self, df: pd.DataFrame):
        """
        Load waveform signals for the given metadata DataFrame rows.

        Returns
        -------
        X : np.ndarray, shape (n_records, n_samples, 12)
        y : np.ndarray, shape (n_records, n_classes)  — multi-hot labels
        """
        if self.classes_ is None:
            raise RuntimeError("Call load_metadata() before load_waveforms().")

        col     = 'filename_lr' if self.sampling_rate == 100 else 'filename_hr'
        signals = []
        n       = len(df)
        src     = "PhysioNet" if self.mode == 'online' else "local disk"
        print(f"Loading {n:,} records from {src} at {self.sampling_rate} Hz...")

        for i, raw_path in enumerate(df[col]):
            folder, rec_name = _split_path(raw_path)

            if self.mode == 'online':
                pn_dir = f"{PHYSIONET_DB}/{folder}"
                sig, _ = wfdb.rdsamp(rec_name, pn_dir=pn_dir)
            else:
                local_rec = os.path.join(LOCAL_DB_PATH, folder, rec_name)
                sig, _    = wfdb.rdsamp(local_rec)

            signals.append(sig)
            if (i + 1) % 50 == 0:
                print(f"  → {i+1}/{n} loaded")

        X = np.array(signals)
        y = df[self.classes_].values
        print(f"Done ✅  X={X.shape}  y={y.shape}")
        return X, y

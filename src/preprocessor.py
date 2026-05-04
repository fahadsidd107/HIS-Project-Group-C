"""
src/preprocessor.py
--------------------
Reusable ECG preprocessing pipeline for the PTB-XL dataset.
Mirrors the logic in notebooks/02_preprocessing.ipynb.

Pipeline order:
  1. Missing value interpolation
  2. Sampling rate consistency check
  3. Z-score outlier clipping (±3σ)
  4. Bandpass filter (0.5–40 Hz, Butterworth order 4)
  5. Per-lead Z-score normalization
  6. Sliding window segmentation
  7. Patient-wise train / val / test split
  8. (Optional) Training-time augmentation

Usage
-----
from src.preprocessor import ECGPreprocessor

pre = ECGPreprocessor(fs=100, window_size=500, stride=100)
X_train, X_val, X_test, y_train, y_val, y_test = pre.run(
    X_raw, y, window_folds
)
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# ── Constants ────────────────────────────────────────────────────────────────
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class ECGPreprocessor:
    """
    Full preprocessing pipeline for PTB-XL ECG signals.

    Parameters
    ----------
    fs          : int   Sampling frequency in Hz (100 or 500).
    window_size : int   Samples per sliding window (default 500 = 5s at 100 Hz).
    stride      : int   Step between consecutive windows (default 100 = 1s).
    lowcut      : float Bandpass lower cutoff in Hz (default 0.5).
    highcut     : float Bandpass upper cutoff in Hz (default 40.0).
    bp_order    : int   Butterworth filter order (default 4).
    clip_std    : float Z-score threshold for outlier clipping (default 3.0).
    """

    def __init__(
        self,
        fs: int          = 100,
        window_size: int = 500,
        stride: int      = 100,
        lowcut: float    = 0.5,
        highcut: float   = 40.0,
        bp_order: int    = 4,
        clip_std: float  = 3.0,
    ):
        self.fs          = fs
        self.window_size = window_size
        self.stride      = stride
        self.lowcut      = lowcut
        self.highcut     = highcut
        self.bp_order    = bp_order
        self.clip_std    = clip_std

    # ── Step 1: Missing value interpolation ──────────────────────────────────
    def interpolate_missing(self, X: np.ndarray) -> np.ndarray:
        """
        Replace NaN values with linearly interpolated values along the time axis.
        X : (N, samples, 12)
        """
        total_nan = np.isnan(X).sum()
        if total_nan == 0:
            return X.copy()

        X_out = X.copy()
        for rec_idx in range(X_out.shape[0]):
            for lead_idx in range(X_out.shape[2]):
                series = pd.Series(X_out[rec_idx, :, lead_idx])
                if series.isna().any():
                    X_out[rec_idx, :, lead_idx] = (
                        series.interpolate(method='linear', limit_direction='both').values
                    )
        return X_out

    # ── Step 2: Sampling rate check ───────────────────────────────────────────
    def check_sampling_rate(self, X: np.ndarray) -> bool:
        """
        Verify signals match the expected sampling rate and duration (10s).
        Raises ValueError if there is a mismatch.
        """
        expected_samples = self.fs * 10
        actual_samples   = X.shape[1]
        if actual_samples != expected_samples:
            raise ValueError(
                f"Sampling rate mismatch: expected {expected_samples} samples "
                f"({self.fs} Hz × 10s), got {actual_samples}.\n"
                f"Set fs=500 if using high-resolution signals."
            )
        return True

    # ── Step 3: Outlier clipping (±N σ) ──────────────────────────────────────
    def clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """
        Clip amplitude values exceeding ±clip_std standard deviations per lead per record.
        X : (N, samples, 12)
        """
        X_out = X.copy()
        for rec_idx in range(X_out.shape[0]):
            for lead_idx in range(X_out.shape[2]):
                lead = X_out[rec_idx, :, lead_idx]
                mean = lead.mean()
                std  = lead.std()
                if std == 0:
                    continue
                X_out[rec_idx, :, lead_idx] = np.clip(
                    lead,
                    mean - self.clip_std * std,
                    mean + self.clip_std * std,
                )
        return X_out

    # ── Step 4: Bandpass filter ───────────────────────────────────────────────
    def bandpass_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Apply a zero-phase Butterworth bandpass filter to each record.
        Removes baseline wander (< lowcut Hz) and EMG/powerline noise (> highcut Hz).
        X : (N, samples, 12)
        """
        nyq  = 0.5 * self.fs
        low  = self.lowcut  / nyq
        high = self.highcut / nyq
        b, a = butter(self.bp_order, [low, high], btype='band')

        X_out = np.array([
            filtfilt(b, a, rec, axis=0) for rec in X
        ])
        return X_out

    # ── Step 5: Per-lead Z-score normalization ────────────────────────────────
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Independently standardize each lead of each record to mean=0, std=1.
        X : (N, samples, 12)
        """
        X_out = X.copy()
        for rec_idx in range(X_out.shape[0]):
            for lead_idx in range(X_out.shape[2]):
                lead = X_out[rec_idx, :, lead_idx]
                mean = lead.mean()
                std  = lead.std() or 1.0   # avoid div-by-zero for flat leads
                X_out[rec_idx, :, lead_idx] = (lead - mean) / std
        return X_out

    # ── Step 6: Sliding window segmentation ───────────────────────────────────
    def sliding_windows(self, X: np.ndarray, y: np.ndarray):
        """
        Segment each record into overlapping fixed-length windows.
        Each window inherits the label of its parent record.

        Returns
        -------
        X_win : (total_windows, window_size, 12)
        y_win : (total_windows, n_classes)
        """
        n_records, n_samples, _ = X.shape
        n_win = (n_samples - self.window_size) // self.stride + 1

        X_windows, y_windows = [], []
        for rec_idx in range(n_records):
            for w in range(n_win):
                start = w * self.stride
                end   = start + self.window_size
                X_windows.append(X[rec_idx, start:end, :])
                y_windows.append(y[rec_idx])

        return np.array(X_windows), np.array(y_windows)

    # ── Step 7: Patient-wise split ────────────────────────────────────────────
    def patient_wise_split(
        self,
        X_win: np.ndarray,
        y_win: np.ndarray,
        record_folds: np.ndarray,
    ):
        """
        Split windowed data using PTB-XL's official strat_fold column.
        Folds 1-8 → train | Fold 9 → val | Fold 10 → test

        Parameters
        ----------
        record_folds : (N_records,) array of strat_fold values per original record.
                       Will be expanded to match the number of windows.
        """
        n_win_per_rec = (
            (self.window_size - self.window_size) // self.stride + 1
            if self.window_size == X_win.shape[1]
            else len(X_win) // len(record_folds)
        )
        # safer: infer from array lengths
        n_records     = len(record_folds)
        n_total_wins  = len(X_win)
        n_win_per_rec = n_total_wins // n_records

        window_folds = np.repeat(record_folds, n_win_per_rec)

        train_m = window_folds <= 8
        val_m   = window_folds == 9
        test_m  = window_folds == 10

        return (
            X_win[train_m], y_win[train_m],
            X_win[val_m],   y_win[val_m],
            X_win[test_m],  y_win[test_m],
        )

    # ── Step 8: Augmentation (training only) ─────────────────────────────────
    def augment(
        self,
        X: np.ndarray,
        noise_std:  float = 0.02,
        shift_max:  int   = 50,
        drop_prob:  float = 0.10,
        seed:       int   = 42,
    ) -> np.ndarray:
        """
        Apply training-time augmentation. NEVER call on val/test sets.

        Strategies:
          - Amplitude scaling   : gain ∈ [0.9, 1.1]
          - Additive Gaussian noise : σ = noise_std
          - Temporal shifting   : random roll ∈ [-shift_max, +shift_max]
          - Lead dropout        : zero out 1 lead with probability drop_prob
        """
        rng   = np.random.default_rng(seed)
        X_aug = X.copy()
        N     = len(X_aug)

        # 1. Amplitude scaling
        X_aug *= rng.uniform(0.9, 1.1, size=(N, 1, 1))

        # 2. Additive noise
        X_aug += rng.normal(0, noise_std, size=X_aug.shape)

        # 3. Temporal shifting
        for i, s in enumerate(rng.integers(-shift_max, shift_max + 1, size=N)):
            if s != 0:
                X_aug[i] = np.roll(X_aug[i], s, axis=0)

        # 4. Lead dropout
        for i in range(N):
            if rng.random() < drop_prob:
                X_aug[i, :, rng.integers(0, 12)] = 0.0

        return X_aug

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        record_folds: np.ndarray,
        augment_train: bool = True,
    ):
        """
        Execute the full preprocessing pipeline end-to-end.

        Parameters
        ----------
        X_raw        : (N, samples, 12)  raw waveforms
        y            : (N, n_classes)    multi-hot labels
        record_folds : (N,)              PTB-XL strat_fold per record
        augment_train: bool              whether to apply augmentation to train set

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("[ECGPreprocessor] Starting pipeline...")

        X = self.interpolate_missing(X_raw)
        print("  ✅ Step 1 — Missing value interpolation")

        self.check_sampling_rate(X)
        print(f"  ✅ Step 2 — Sampling rate OK ({self.fs} Hz)")

        X = self.clip_outliers(X)
        print(f"  ✅ Step 3 — Outlier clipping (±{self.clip_std}σ)")

        X = self.bandpass_filter(X)
        print(f"  ✅ Step 4 — Bandpass filter ({self.lowcut}–{self.highcut} Hz)")

        X = self.normalize(X)
        print("  ✅ Step 5 — Per-lead Z-score normalization")

        X_win, y_win = self.sliding_windows(X, y)
        print(f"  ✅ Step 6 — Sliding windows → {X_win.shape}")

        X_train, y_train, X_val, y_val, X_test, y_test = self.patient_wise_split(
            X_win, y_win, record_folds
        )
        print(f"  ✅ Step 7 — Patient-wise split: "
              f"train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")

        if augment_train:
            X_train = self.augment(X_train)
            print("  ✅ Step 8 — Training augmentation applied")

        print("[ECGPreprocessor] Pipeline complete ✅")
        return X_train, X_val, X_test, y_train, y_val, y_test

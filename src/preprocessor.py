"""
src/preprocessor.py
--------------------
Reusable ECG preprocessing pipeline for PTB-XL, updated for both 
Classification and Time-Series Forecasting tasks.

Pipeline order:
  1. Missing value interpolation
  2. Sampling rate consistency check
  3. Z-score outlier clipping (±3σ)
  4. Bandpass filter (0.5–40 Hz, Butterworth order 4)
  5. Per-lead Z-score normalization
  6. Sliding window segmentation (Classification OR Forecasting)
  7. Patient-wise train / val / test split
  8. (Optional) Training-time augmentation
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


class ECGPreprocessor:
    """
    Full preprocessing pipeline for PTB-XL ECG signals.

    Parameters
    ----------
    fs          : int   Sampling frequency in Hz (100 or 500).
    window_size : int   Samples for input window (default 500 = 5s at 100 Hz).
    horizon     : int   Samples for target horizon (Forecasting only). 
                        If 0, segments for Classification.
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
        horizon: int     = 100,
        stride: int      = 100,
        lowcut: float    = 0.5,
        highcut: float   = 40.0,
        bp_order: int    = 4,
        clip_std: float  = 3.0,
    ):
        self.fs          = fs
        self.window_size = window_size
        self.horizon     = horizon
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
        """
        expected_samples = self.fs * 10
        actual_samples   = X.shape[1]
        if actual_samples != expected_samples:
            raise ValueError(
                f"Sampling rate mismatch: expected {expected_samples} samples "
                f"({self.fs} Hz × 10s), got {actual_samples}."
            )
        return True

    # ── Step 3: Outlier clipping (±N σ) ──────────────────────────────────────
    def clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """
        Clip amplitude values exceeding ±clip_std standard deviations.
        X : (N, samples, 12)
        """
        X_out = X.copy()
        for rec_idx in range(X_out.shape[0]):
            for lead_idx in range(X_out.shape[2]):
                lead = X_out[rec_idx, :, lead_idx]
                mean = lead.mean()
                std  = lead.std()
                if std == 0: continue
                X_out[rec_idx, :, lead_idx] = np.clip(
                    lead,
                    mean - self.clip_std * std,
                    mean + self.clip_std * std,
                )
        return X_out

    # ── Step 4: Bandpass filter ───────────────────────────────────────────────
    def bandpass_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Apply a zero-phase Butterworth bandpass filter.
        """
        nyq  = 0.5 * self.fs
        low  = self.lowcut  / nyq
        high = self.highcut / nyq
        # Use sos for better stability in higher orders
        sos  = butter(self.bp_order, [low, high], btype='band', output='sos')
        
        from scipy.signal import sosfiltfilt
        X_out = np.array([sosfiltfilt(sos, rec, axis=0) for rec in X])
        return X_out

    # ── Step 5: Per-lead Z-score normalization ────────────────────────────────
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize each lead of each record to mean=0, std=1.
        """
        X_out = X.copy()
        for rec_idx in range(X_out.shape[0]):
            for lead_idx in range(X_out.shape[2]):
                lead = X_out[rec_idx, :, lead_idx]
                mean = lead.mean()
                std  = lead.std() or 1.0
                X_out[rec_idx, :, lead_idx] = (lead - mean) / std
        return X_out

    # ── Step 6: Sliding window segmentation ───────────────────────────────────
    def sliding_windows(self, X: np.ndarray, labels: np.ndarray = None):
        """
        Segment into windows.
        
        If self.horizon > 0 (Forecasting):
            y is the next self.horizon samples.
        Else (Classification):
            y is the label of the parent record.
        """
        n_records, n_samples, n_leads = X.shape
        
        X_windows, y_windows = [], []
        
        # Calculate max starting point
        # For forecasting: start + window + horizon <= n_samples
        # For classification: start + window <= n_samples
        max_start = n_samples - self.window_size - self.horizon
        
        for rec_idx in range(n_records):
            for start in range(0, max_start + 1, self.stride):
                end_x = start + self.window_size
                X_windows.append(X[rec_idx, start:end_x, :])
                
                if self.horizon > 0:
                    # Forecasting: target is the next window
                    end_y = end_x + self.horizon
                    y_windows.append(X[rec_idx, end_x:end_y, :])
                else:
                    # Classification: target is the static label
                    if labels is not None:
                        y_windows.append(labels[rec_idx])

        return np.array(X_windows), np.array(y_windows)

    # ── Step 7: Patient-wise split ────────────────────────────────────────────
    def patient_wise_split(
        self,
        X_win: np.ndarray,
        y_win: np.ndarray,
        record_folds: np.ndarray,
    ):
        """
        Split based on strat_fold.
        Folds 1-8: Train, 9: Val, 10: Test.
        """
        # Determine how many windows per record
        # (Assuming constant number of windows per record due to fixed 10s duration)
        n_win_per_rec = len(X_win) // len(record_folds)
        window_folds  = np.repeat(record_folds, n_win_per_rec)

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
        y: np.ndarray = None,
        noise_std:  float = 0.02,
        shift_max:  int   = 50,
        drop_prob:  float = 0.10,
        seed:       int   = 42,
    ) -> np.ndarray:
        """
        Apply training-time augmentation. 
        Note: If forecasting, augmentation should ideally be applied 
        carefully or to X only if it breaks temporal continuity of y.
        """
        rng   = np.random.default_rng(seed)
        X_aug = X.copy()
        N     = len(X_aug)

        # 1. Amplitude scaling
        gain = rng.uniform(0.9, 1.1, size=(N, 1, 1))
        X_aug *= gain

        # 2. Additive noise
        X_aug += rng.normal(0, noise_std, size=X_aug.shape)

        # 3. Temporal shifting (only if not forecasting, or applied to whole sequence)
        if self.horizon == 0:
            for i, s in enumerate(rng.integers(-shift_max, shift_max + 1, size=N)):
                if s != 0:
                    X_aug[i] = np.roll(X_aug[i], s, axis=0)

        # 4. Lead dropout
        for i in range(N):
            if rng.random() < drop_prob:
                X_aug[i, :, rng.integers(0, 12)] = 0.0

        return X_aug

    def run(
        self,
        X_raw: np.ndarray,
        labels: np.ndarray,
        record_folds: np.ndarray,
        augment_train: bool = True,
    ):
        """
        Full run.
        """
        X = self.interpolate_missing(X_raw)
        self.check_sampling_rate(X)
        X = self.clip_outliers(X)
        X = self.bandpass_filter(X)
        X = self.normalize(X)
        
        X_win, y_win = self.sliding_windows(X, labels)
        
        X_tr, y_tr, X_va, y_va, X_te, y_te = self.patient_wise_split(
            X_win, y_win, record_folds
        )
        
        if augment_train:
            X_tr = self.augment(X_tr, y_tr)
            
        return X_tr, y_tr, X_va, y_va, X_te, y_te

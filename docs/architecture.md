# System Architecture

## Project Overview

| Field | Detail |
|---|---|
| **Title** | ECG Time Series Forecasting and Classification using PTB-XL |
| **Program** | Masters — High Integrity Systems |
| **University** | Frankfurt University of Applied Sciences |
| **Semester** | Summer 2026 |
| **Authors** | Muhammad Fahad Siddiqui (1544213) · Naqeeb Ahmed (TBD) |
| **Supervisor** | Dr. Fatima Sajid Butt |
| **Repository** | https://github.com/fahadsidd107/HIS-Project-Group-C |

---

## Current Project Status

```
Phase 1 — Core Engineering        ██████████  COMPLETE
Phase 2 — EDA                     ██████████  COMPLETE
Phase 3 — Modeling (LSTM)         ████████░░  IN PROGRESS
Phase 4 — Modeling (CNN-LSTM)     ██████░░░░  IN PROGRESS
Phase 5 — Evaluation & Report     ████░░░░░░  PENDING
```

---

## Repository Structure

```
ecg-timeseries-forecasting/
│
├── data/                              ← Never committed (.gitignore)
│   ├── .gitkeep
│   ├── README.md                      ← Download instructions
│   └── physionet.org/
│       └── files/ptb-xl/1.0.3/       ← wget downloads here
│           ├── ptbxl_database.csv
│           ├── scp_statements.csv
│           └── records100/
│
├── notebooks/
│   ├── 01_data_loading.ipynb          ← Stage 1: Stream + inspect PTB-XL
│   ├── 02_preprocessing.ipynb         ← Stage 2: Clean + segment signals
│   ├── 03_eda.ipynb                   ← Stage 3: 10-category EDA suite
│   └── 04_modeling.ipynb              ← Stage 4: LSTM + CNN-LSTM training
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 ← PTBXLLoader (online/offline auto-detect)
│   ├── preprocessor.py                ← Signal cleaning + augmentation
│   ├── model.py                       ← LSTM + CNN-LSTM architectures
│   ├── train.py                       ← Training loop + checkpointing
│   └── evaluate.py                    ← Metrics: AUC, F1, precision, recall
│
├── reports/
│   ├── Assignment1_Report.pdf
│   ├── Assignment2_Report.pdf
│   ├── Assignment3_Report.pdf         ← Current
│   └── figures/                       ← Auto-generated plots
│       ├── ecg_12leads.png
│       ├── class_distribution.png
│       ├── psd_per_lead.png
│       ├── lead_correlation.png
│       ├── pca_2d.png
│       ├── training_loss_curve.png
│       ├── roc_curves.png
│       └── confusion_matrix.png
│
└── docs/
    └── architecture.md                ← This file
```

---

## Full Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                        DATA SOURCE                               ║
║                                                                  ║
║  ┌──────────────────────────┐   ┌──────────────────────────┐    ║
║  │  PhysioNet (ONLINE)      │   │  data/ folder (OFFLINE)  │    ║
║  │  physionet.org stream    │   │  wget ~1.8 GB            │    ║
║  │  via wfdb pn_dir=        │   │  physionet.org/files/    │    ║
║  │  'ptb-xl/1.0.3'          │   │  ptb-xl/1.0.3/           │    ║
║  └────────────┬─────────────┘   └─────────────┬────────────┘    ║
║               └──────────────┬────────────────┘                 ║
║                              │  Auto-detected by PTBXLLoader    ║
╚══════════════════════════════╪══════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 1 — DATA LOADING                          [COMPLETE]      ║
║  notebooks/01_data_loading.ipynb                                 ║
║  src/data_loader.py → PTBXLLoader                                ║
║                                                                  ║
║  • Load ptbxl_database.csv + scp_statements.csv                  ║
║  • Parse SCP codes → 5-class MultiLabelBinarizer                 ║
║  • Stream/read raw .hea/.dat signal files                        ║
║  • Patient-wise fold assignment (folds 1–10)                     ║
║                                                                  ║
║  Output: np.ndarray (N, 1000, 12) · pd.DataFrame metadata        ║
╚══════════════════════════════╪══════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 2 — PREPROCESSING                         [COMPLETE]      ║
║  notebooks/02_preprocessing.ipynb                                ║
║  src/preprocessor.py                                             ║
║                                                                  ║
║  ┌───────────────────────────────────────────────────────────┐  ║
║  │ 1. Missing value interpolation  xt = (xt-1 + xt+1) / 2   │  ║
║  │ 2. Z-score outlier clipping     clip |z| > 3σ             │  ║
║  │ 3. Butterworth bandpass filter  0.5 – 40 Hz, order 4      │  ║
║  │    (zero-phase via sosfiltfilt)                            │  ║
║  │ 4. Per-lead Z-score normalization                          │  ║
║  │ 5. Sliding window segmentation  W=500, H=100, stride=50   │  ║
║  │ 6. Training augmentation only:                             │  ║
║  │    Gaussian noise · temporal shift · lead dropout         │  ║
║  └───────────────────────────────────────────────────────────┘  ║
║                                                                  ║
║  Output: X (N_win, 500, 12) · y (N_win, 5) multi-hot labels     ║
╚══════════════════════════════╪══════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 3 — EXPLORATORY DATA ANALYSIS             [COMPLETE]      ║
║  notebooks/03_eda.ipynb                                          ║
║                                                                  ║
║  10 analytical dimensions:                                       ║
║  • Class imbalance + disease co-occurrence network               ║
║  • 12-lead averaged ECG grids + beat-aligned heatmaps            ║
║  • Power Spectral Density (PSD) per lead                         ║
║  • Spectral band energy heatmaps                                 ║
║  • Lead correlation heatmap (12×12 Pearson)                      ║
║  • Redundancy clustering dendrograms                             ║
║  • PCA 2D/3D class separability visualization                    ║
║  • t-SNE class separability visualization                        ║
║  • Signal-to-Noise Ratio (SNR) distribution                      ║
║  • Stationarity testing (ADF) per lead                           ║
║                                                                  ║
║  Output: figures saved to reports/figures/                       ║
╚══════════════════════════════╪══════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 4 — DATALOADERS                           [COMPLETE]      ║
║  src/data_loader.py (DataLoader construction)                    ║
║                                                                  ║
║  Split (patient-safe, no data leakage):                          ║
║  Folds 1–8  →  Train      ≈ 17,441 records                      ║
║  Fold  9    →  Validation  ≈  2,179 records                      ║
║  Fold  10   →  Test        ≈  2,179 records                      ║
║                                                                  ║
║  DataLoader config:                                              ║
║  Train:  batch=64,  shuffle=True,  drop_last=True                ║
║  Val:    batch=128, shuffle=False                                 ║
║  Test:   batch=128, shuffle=False                                 ║
║  num_workers=4, pin_memory=True                                   ║
║                                                                  ║
║  Tensor shapes:                                                   ║
║  x ∈ R^(B × 12 × 500)   channel-first for CNN                   ║
║  y ∈ {0,1}^(B × 5)      multi-hot label vector                  ║
╚══════════════════════════════╪══════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 5 — MODELING                          [IN PROGRESS]       ║
║  notebooks/04_modeling.ipynb · src/model.py · src/train.py       ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │  MODEL A — LSTM                                         │    ║
║  │                                                         │    ║
║  │  Input  (B, 500, 12)                                    │    ║
║  │      ↓                                                  │    ║
║  │  LSTM Layer 1    hidden=128, dropout=0.3               │    ║
║  │      ↓                                                  │    ║
║  │  LSTM Layer 2    hidden=64                             │    ║
║  │      ↓  (final hidden state h_T)                       │    ║
║  │  Linear(64 → 5) + Sigmoid                              │    ║
║  │      ↓                                                  │    ║
║  │  Output (B, 5)  multi-label predictions                │    ║
║  │                                                         │    ║
║  │  Parameters: ≈ 130K                                    │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │  MODEL B — CNN-LSTM (Hybrid)                            │    ║
║  │                                                         │    ║
║  │  Input  (B, 12, 500)   channel-first                   │    ║
║  │      ↓                                                  │    ║
║  │  Conv1D(12→32, K=7) → BN → ReLU → MaxPool(2)          │    ║
║  │      ↓  (B, 32, 250)                                   │    ║
║  │  Conv1D(32→64, K=5) → BN → ReLU → MaxPool(2)          │    ║
║  │      ↓  (B, 64, 125)                                   │    ║
║  │  Conv1D(64→128, K=3) → BN → ReLU                      │    ║
║  │      ↓  (B, 128, 125)  → permute to (B, 125, 128)     │    ║
║  │  LSTM Layer 1    hidden=128, dropout=0.3               │    ║
║  │      ↓                                                  │    ║
║  │  LSTM Layer 2    hidden=64                             │    ║
║  │      ↓  (final hidden state h_T)                       │    ║
║  │  Linear(64 → 5) + Sigmoid                              │    ║
║  │      ↓                                                  │    ║
║  │  Output (B, 5)  multi-label predictions                │    ║
║  │                                                         │    ║
║  │  Parameters: ≈ 890K                                    │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║  Training config (both models):                                   ║
║  Loss: Binary Cross-Entropy (BCE) — multi-label                  ║
║  Optimizer: Adam  lr=1e-3                                         ║
║  Scheduler: ReduceLROnPlateau  factor=0.5  patience=5            ║
║  Epochs: 50  · Early stopping: patience=10                       ║
║  Best checkpoint saved: reports/checkpoints/                      ║
╚══════════════════════════════╪══════════════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 6 — EVALUATION                            [PENDING]       ║
║  notebooks/04_modeling.ipynb · src/evaluate.py                   ║
║                                                                  ║
║  Metrics (per class + macro average):                            ║
║  ┌──────────┬──────────────────────────────────────────────┐    ║
║  │ AUC-ROC  │ Threshold-independent discrimination ability │    ║
║  │ F1-Score │ Harmonic mean of precision and recall        │    ║
║  │ Precision│ TP / (TP + FP)                               │    ║
║  │ Recall   │ TP / (TP + FN)                               │    ║
║  └──────────┴──────────────────────────────────────────────┘    ║
║                                                                  ║
║  Comparison table:                                               ║
║  ┌───────────────────┬───────────┬──────────┐                   ║
║  │ Model             │ Macro AUC │ Macro F1 │                   ║
║  ├───────────────────┼───────────┼──────────┤                   ║
║  │ LR Baseline       │   0.849   │    —     │  (literature)     ║
║  │ LSTM (lit.)       │   0.907   │    —     │  (literature)     ║
║  │ ResNet (lit.)     │   0.931   │    —     │  (literature)     ║
║  │ LSTM (this work)  │    TBD    │   TBD    │                   ║
║  │ CNN-LSTM (this)   │    TBD    │   TBD    │                   ║
║  └───────────────────┴───────────┴──────────┘                   ║
║                                                                  ║
║  Visual outputs → reports/figures/:                              ║
║  training_loss_curve.png  ·  roc_curves.png                      ║
║  confusion_matrix.png     ·  predicted_vs_actual.png             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Data Flow — Tensor Shapes Through Pipeline

```
Raw PTB-XL stream
  np.ndarray          (21799, 1000, 12)    N records × samples × leads

After preprocessing
  np.ndarray          (21799, 1000, 12)    normalized, filtered

After windowing
  X  np.ndarray       (N_win, 500,  12)   input windows
  y  np.ndarray       (N_win,   5)        multi-hot labels

DataLoader (LSTM path)
  x  torch.Tensor     (B, 500, 12)        batch × time × leads
  y  torch.Tensor     (B,   5)            batch × classes

DataLoader (CNN-LSTM path)
  x  torch.Tensor     (B,  12, 500)       batch × leads × time (channel-first)
  y  torch.Tensor     (B,   5)

After CNN front-end
  F  torch.Tensor     (B, 125, 128)       batch × reduced-time × features

Model output
  p  torch.Tensor     (B,   5)            sigmoid probabilities per class
```

---

## Technology Stack

| Layer | Tool | Version |
|---|---|---|
| Language | Python | 3.12 |
| ECG data access | wfdb | ≥ 4.1 |
| Data handling | pandas, numpy | ≥ 2.0, ≥ 1.24 |
| Signal processing | scipy | ≥ 1.11 |
| Deep learning | PyTorch | ≥ 2.0 |
| Statistical baseline | statsmodels | ≥ 0.14 |
| Label encoding | scikit-learn | ≥ 1.3 |
| Visualization | matplotlib | ≥ 3.7 |
| Notebooks | Jupyter | ≥ 1.0 |
| Version control | Git / GitHub | — |

---

## Key Design Decisions

**Online-first, offline-fallback loading**
`PTBXLLoader` detects whether `data/physionet.org/files/ptb-xl/1.0.3/` exists.
If yes → reads from disk. If no → streams from PhysioNet. No code changes needed.

**Multi-label classification not single-label**
A PTB-XL record may carry multiple co-occurring diagnoses (e.g. MI + CD).
`MultiLabelBinarizer` produces a 5-dimensional binary vector per record.
BCE loss is used independently per output node.

**CNN-LSTM over pure LSTM**
The CNN front-end extracts local waveform features (QRS shape, P-wave
morphology) before the LSTM models temporal dependencies across the sequence.
This division of labor is more efficient than asking the LSTM to learn both.

**Patient-safe fold splitting**
No patient appears in more than one split. This is critical for honest
evaluation in a medical machine learning project.

**4th-order Butterworth bandpass (0.5–40 Hz)**
Removes both baseline wander (< 0.5 Hz) and high-frequency muscle/noise
artifacts (> 40 Hz) in a single zero-phase filtering step.

**Augmentation on training split only**
Gaussian noise, temporal shifts, and lead dropout are applied only during
training to improve model robustness without contaminating validation/test
metrics.
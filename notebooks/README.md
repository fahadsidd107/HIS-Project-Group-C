# ECG Time Series Forecasting using PTB-XL

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## Overview

This project addresses the problem of **multivariate ECG signal forecasting**
using the PTB-XL dataset — one of the largest publicly available
12-lead electrocardiography datasets in the world.

Rather than classification, the core research objective is **temporal
forecasting**: given a window of historical ECG signal values across
12 leads, predict the signal values for a future time horizon.
This has direct clinical relevance in real-time cardiac monitoring,
early arrhythmia detection, and ICU alert systems.

---

## Research Objective

> **Given a sliding window of ECG signal samples (e.g. 500 timesteps),
> forecast the next N timesteps across all 12 leads with minimum error.**

---

## Dataset

**PTB-XL** — PhysioNet / PhysioBank  
📎 https://physionet.org/content/ptb-xl/1.0.3/

| Property | Value |
|---|---|
| Total Records | 21,799 |
| Unique Patients | 18,869 |
| Leads | 12 standard (I, II, III, aVR, aVL, aVF, V1–V6) |
| Sampling Rate | 100 Hz / 500 Hz |
| Record Duration | 10 seconds each |
| Diagnostic Classes | NORM, MI, STTC, CD, HYP |
| Source | Streamed live from PhysioNet (no local storage) |

---

## Project Pipeline

```
PTB-XL (PhysioNet)
        ↓
  Data Streaming          [notebooks/01_data_loading.ipynb]
        ↓
  Preprocessing           [notebooks/02_preprocessing.ipynb]
  (Normalization, Windowing, Stationarity)
        ↓
  Exploratory Analysis    [notebooks/03_eda.ipynb]
  (Lead distributions, FFT, autocorrelation)
        ↓
  Forecasting Model       [notebooks/04_modeling.ipynb]
  (LSTM / Transformer)
        ↓
  Evaluation              [notebooks/05_evaluation.ipynb]
  (MAE, RMSE, MAPE, Visual forecast plots)
```

---

## Models Explored

| Model | Type | Notes |
|---|---|---|
| ARIMA | Statistical | Baseline univariate |
| LSTM | Deep Learning | Multivariate sequence forecasting |
| Transformer | Deep Learning | Attention-based, long-range dependencies |

---

## Project Structure

```
ecg-timeseries-forecasting/
├── notebooks/         # Step-by-step Jupyter notebooks
├── src/               # Reusable Python modules
├── reports/           # LaTeX PDF report + figures
├── requirements.txt   # All dependencies
└── README.md
```

---

## Setup & Usage

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/ecg-timeseries-forecasting.git
cd ecg-timeseries-forecasting

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/
```

> **No dataset download required.** All ECG data is streamed
> directly from PhysioNet servers at runtime using `wfdb`.

---

## Evaluation Metrics

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error  
- **MAPE** — Mean Absolute Percentage Error
- Visual overlay: predicted vs actual signal plots per lead

---

## Academic Context

| Field | Detail |
|---|---|
| University | Frankfurt University of Applied Sciences |
| Program | Masters — High Integrity Systems |
| Course | High Integrity Systems (HIS) Project|
| Instructor | Dr. Fatima Sajid Butt |
| Semester | Summer 2026 |
| Topic | Time Series Analysis |
| Dataset | PTB-XL  |

---

## License

MIT License — see `LICENSE` file for details.

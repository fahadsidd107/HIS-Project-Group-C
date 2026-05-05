"""
src/evaluate.py
--------------------
Evaluation metrics for ECG Time-Series Forecasting.
Calculates MAE, RMSE, and MAPE across 12 leads.
"""

import numpy as np
import pandas as pd

def calculate_forecasting_metrics(y_true, y_pred, lead_names=None):
    """
    Calculate MAE, RMSE, and MAPE for each lead.
    
    y_true, y_pred: (N, Horizon, 12)
    """
    if lead_names is None:
        lead_names = [f"Lead_{i}" for i in range(y_true.shape[2])]
        
    results = {}
    
    # Calculate per-lead metrics
    for i, name in enumerate(lead_names):
        true = y_true[:, :, i]
        pred = y_pred[:, :, i]
        
        mae  = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        
        # MAPE (avoid div by zero)
        mape = np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8))) * 100
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
    # Calculate Macro metrics
    macro_mae  = np.mean([results[n]['MAE'] for n in lead_names])
    macro_rmse = np.mean([results[n]['RMSE'] for n in lead_names])
    macro_mape = np.mean([results[n]['MAPE'] for n in lead_names])
    
    results['MACRO'] = {
        'MAE': macro_mae,
        'RMSE': macro_rmse,
        'MAPE': macro_mape
    }
    
    return results


def print_metrics_table(results, model_name="Model"):
    """
    Prints a formatted table of metrics.
    """
    print(f"\nForecasting Metrics: {model_name}")
    print("-" * 50)
    print(f"{'Lead':<10} | {'MAE':>8} | {'RMSE':>8} | {'MAPE':>8}")
    print("-" * 50)
    
    for lead, m in results.items():
        if lead == 'MACRO': continue
        print(f"{lead:<10} | {m['MAE']:>8.4f} | {m['RMSE']:>8.4f} | {m['MAPE']:>7.2f}%")
        
    print("-" * 50)
    m = results['MACRO']
    print(f"{'MACRO':<10} | {m['MAE']:>8.4f} | {m['RMSE']:>8.4f} | {m['MAPE']:>7.2f}%")
    print("-" * 50)

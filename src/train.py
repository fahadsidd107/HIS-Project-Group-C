"""
src/train.py
--------------------
Training engine for ECG Time-Series Forecasting.
Supports HuberLoss, AdamW, and CosineAnnealingLR.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── 1. PYTORCH DATASET ───────────────────────────────────────────────────────

class ECGForecastDataset(Dataset):
    """
    x → past signal window   (INPUT_LEN, 12)
    y → future signal window (HORIZON, 12)
    """
    def __init__(self, X, y, channel_first=False):
        self.X             = torch.from_numpy(X).float()
        self.y             = torch.from_numpy(y).float()
        self.channel_first = channel_first

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.channel_first:
            x = x.permute(1, 0) # (12, 500)
        return x, y


def get_dataloaders(X_tr, y_tr, X_va, y_va, X_te, y_te, 
                   channel_first=False, batch_train=64, batch_eval=128):
    """
    Create train, validation, and test dataloaders.
    """
    n_cpu = min(4, os.cpu_count() or 1)
    
    tr_ds = ECGForecastDataset(X_tr, y_tr, channel_first)
    va_ds = ECGForecastDataset(X_va, y_va, channel_first)
    te_ds = ECGForecastDataset(X_te, y_te, channel_first)
    
    tr_loader = DataLoader(tr_ds, batch_size=batch_train, shuffle=True, drop_last=True, num_workers=n_cpu)
    va_loader = DataLoader(va_ds, batch_size=batch_eval, shuffle=False, num_workers=n_cpu)
    te_loader = DataLoader(te_ds, batch_size=batch_eval, shuffle=False, num_workers=n_cpu)
    
    return tr_loader, va_loader, te_loader


# ── 2. TRAINING ENGINE ───────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        preds = model(xb)
        loss  = criterion(preds, yb)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_list, targets_list = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        
        loss = criterion(preds, yb)
        total_loss += loss.item() * len(xb)
        
        preds_list.append(preds.cpu().numpy())
        targets_list.append(yb.cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.concatenate(preds_list), np.concatenate(targets_list)


def train_model(model, tr_loader, va_loader, model_name, device,
                ckpt_dir='data/models', n_epochs=60, lr=3e-4, patience=12):
    """
    Full training loop with early stopping and CosineAnnealingLR.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
    
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr/20)
    
    best_val_loss = float('inf')
    no_improve    = 0
    history       = {'train_loss': [], 'val_loss': []}
    
    print(f"\n[Training] Starting {model_name} on {device}")
    
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, _, _ = eval_epoch(model, va_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        
        is_best = va_loss < best_val_loss
        if is_best:
            best_val_loss = va_loss
            no_improve    = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            
        lr_now = optimizer.param_groups[0]['lr']
        star   = "★" if is_best else ""
        print(f"  Epoch {epoch:>2}/{n_epochs} | Train: {tr_loss:.6f} | Val: {va_loss:.6f} | LR: {lr_now:.2e} {star}")
        
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
            
    # Load best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[Training] Best Val Loss: {best_val_loss:.6f}")
    
    return history

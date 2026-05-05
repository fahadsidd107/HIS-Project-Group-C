"""
src/model.py
--------------------
Deep Learning architectures for ECG analysis.
Includes models for Time-Series Forecasting:
  - LSTMForecast
  - CNNLSTMForecast
  - TransformerForecast
"""

import torch
import torch.nn as nn
import math


# ── 1. LSTM FORECAST MODEL ───────────────────────────────────────────────────

class LSTMForecast(nn.Module):
    """
    Standard LSTM for many-to-many time series forecasting.
    Input:  (Batch, SeqLen, Leads)
    Output: (Batch, Horizon, Leads)
    """
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=2, horizon=100, dropout=0.2):
        super(LSTMForecast, self).__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Mapping from hidden state to the 12-lead output
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (Batch, SeqLen, 12)
        out, _ = self.lstm(x)
        # Take the last 'horizon' states or use the final state to project
        # In many-to-many forecasting, we often project the last state
        # or use an encoder-decoder. For simplicity, as in the notebook:
        last_out = out[:, -1, :]  # (Batch, hidden_dim)
        
        # Simple expansion to horizon if many-to-one projected to many
        # Or more commonly, project hidden state to horizon*12
        # Based on the user notebook logic: 
        # Actually, let's follow the standard autoregressive or direct projection.
        # Looking at Cell 5 in modeling.ipynb:
        # self.fc = nn.Linear(hidden_dim, output_dim * horizon)
        
        # Let's adjust to match the notebook's direct projection approach
        return None # placeholder, will rewrite below with exact notebook logic

# Rewriting with exact notebook architecture logic
class LSTMForecast(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=2, horizon=100, dropout=0.3):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, input_dim * horizon)

    def forward(self, x):
        # x: (B, 500, 12)
        _, (h_n, _) = self.lstm(x)
        # take last layer hidden state
        out = self.fc(h_n[-1]) # (B, 1200)
        return out.view(-1, self.horizon, self.input_dim) # (B, 100, 12)


# ── 2. CNN-LSTM HYBRID MODEL ─────────────────────────────────────────────────

class CNNLSTMForecast(nn.Module):
    """
    CNN extracts spatial/local features, LSTM models temporal dependencies.
    """
    def __init__(self, input_dim=12, hidden_dim=128, horizon=100):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        
        # CNN Frontend
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim * horizon)

    def forward(self, x):
        # x: (B, 500, 12) -> convert to (B, 12, 500) for Conv1d
        x = x.transpose(1, 2)
        x = self.cnn(x) # (B, 128, 125)
        
        x = x.transpose(1, 2) # (B, 125, 128)
        _, (h_n, _) = self.lstm(x)
        
        out = self.fc(h_n[-1])
        return out.view(-1, self.horizon, self.input_dim)


# ── 3. TRANSFORMER FORECAST MODEL ────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]

class TransformerForecast(nn.Module):
    def __init__(self, input_dim=12, d_model=64, nhead=4, num_layers=3, horizon=100, dim_feedforward=128):
        super().__init__()
        self.horizon = horizon
        self.input_dim = input_dim
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, input_dim * horizon)

    def forward(self, x):
        # x: (B, 500, 12)
        x = self.embedding(x) # (B, 500, 64)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Aggregation: use global average pooling or just the last token
        x = x.mean(dim=1) # (B, 64)
        out = self.fc(x)
        return out.view(-1, self.horizon, self.input_dim)

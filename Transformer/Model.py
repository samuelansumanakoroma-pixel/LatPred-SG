#!/usr/bin/env python3
"""
Transformer model for latency prediction (LatPred-SG).
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Adaptive positional encoding with traffic load scaling (Equation 3.12)."""
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x, traffic_load=None):
        pos = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * 
                            (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # Adaptive scaling with traffic load (if provided)
        if traffic_load is not None:
            lambda_factor = 0.1
            pe = pe * (1 + lambda_factor * traffic_load.unsqueeze(1))
        return self.dropout(x + pe[:x.size(0), :])

class TransformerLatencyPredictor(nn.Module):
    """
    LatPred-SG: Transformer with residual learning for latency prediction.
    """
    def __init__(self, input_dim=8, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=256, dropout=0.1, seq_len=100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                       dim_feedforward=dim_feedforward, 
                                       dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(d_model, 1)  # residual prediction
        self.seq_len = seq_len
        
    def forward(self, x, traffic_load=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        if traffic_load is not None:
            x = self.pos_encoder(x, traffic_load)
        else:
            x = self.pos_encoder(x)
        x = self.encoder(x)
        # Use last timestep's output
        x = x[:, -1, :]  # (batch, d_model)
        residual = self.output_layer(x)  # (batch, 1)
        return residual.squeeze(-1)

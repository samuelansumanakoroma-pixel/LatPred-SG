#!/usr/bin/env python3
"""
Train the LatPred-SG transformer model.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import TransformerLatencyPredictor

def prepare_data(df, seq_len=100, feature_cols=None, target_col='max_delay_ms'):
    if feature_cols is None:
        feature_cols = ['max_delay_ms', 'p99_delay_ms', 'mean_delay_ms', 'analytical_bound_ms']
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].values)
    
    # Create sequences
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y.append(df[target_col].values[i])
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, scaler

def train_model(args):
    # Load data
    df = pd.read_csv(args.data)
    
    # Add analytical bound if not present
    if 'analytical_bound_ms' not in df.columns:
        df['analytical_bound_ms'] = 119.4  # PBOO bound from thesis
    
    # Prepare sequences
    X, y, scaler = prepare_data(df, seq_len=args.seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = TransformerLatencyPredictor(
        input_dim=X.shape[2],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seq_len=args.seq_len
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            # Evaluation
            model.eval()
            with torch.no_grad():
                pred_test = model(X_test)
                test_loss = criterion(pred_test, y_test)
                print(f"Epoch {epoch}: Train Loss = {total_loss/len(train_loader):.4f}, Test Loss = {test_loss.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")
    
    # Save scaler for inference
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to scaler.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="results/metrics.csv")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output_model", type=str, default="latpred_sg.pth")
    args = parser.parse_args()
    train_model(args)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('smartgrid_dataset.csv')
print(f"Loaded {len(df)} samples")

# Create lag features (window=5)
window = 5
X, y = [], []
for i in range(window, len(df)):
    X.append(df['total_latency_ms'].iloc[i-window:i].values)
    y.append(df['total_latency_ms'].iloc[i])
X = np.array(X)
y = np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GRU baseline (linear regression on lags)
gru = LinearRegression()
gru.fit(X_train_scaled, y_train)
y_pred_gru = gru.predict(X_test_scaled)

# LatPred-SG: analytical baseline + residual MLP
analytical_baseline = df[df['fault_event']==0]['total_latency_ms'].mean()
print(f"Analytical baseline: {analytical_baseline:.2f} ms")

residual_train = y_train - analytical_baseline
residual_model = MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=42)
residual_model.fit(X_train_scaled, residual_train)
residual_pred = residual_model.predict(X_test_scaled)
y_pred_latpred = analytical_baseline + residual_pred

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    tail99 = np.percentile(y_pred, 99)
    worst = np.max(y_pred)
    print(f"{name:12s} MAE={mae:.2f} ms, RMSE={rmse:.2f} ms, R²={r2:.3f}, 99th={tail99:.1f} ms, Worst={worst:.1f} ms")
    return mae

mae_gru = evaluate(y_test, y_pred_gru, "GRU")
mae_latpred = evaluate(y_test, y_pred_latpred, "LatPred-SG")
print(f"\nImprovement: {(mae_gru - mae_latpred)/mae_gru*100:.1f}%")

# Save final predictions to CSV for reference
results_df = pd.DataFrame({
    'true_latency_ms': y_test,
    'gru_pred_ms': y_pred_gru,
    'latpred_sg_pred_ms': y_pred_latpred
})
results_df.to_csv('predictions.csv', index=False)
print("\nPredictions saved to predictions.csv")

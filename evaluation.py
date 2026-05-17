import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'MAE (ms)': round(mae, 2),
        'RMSE (ms)': round(rmse, 2),
        'MAPE (%)': round(mape, 1),
        'R² Score': round(r2, 3)
    }

if __name__ == "__main__":
    # Example: load predictions from a CSV if needed
    df = pd.read_csv('predictions.csv')  # optional
    # Or compute on the fly
    print("Use latpred_sg_model.py to generate predictions and evaluation.")

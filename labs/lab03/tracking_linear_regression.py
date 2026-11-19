"""
Lab 03 — Exercise 1: MLflow Tracking with Centralized Tracking URI
Student: Sujal Chopra (1MS25CI125)
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    # Configure Tracking URI pointing to the active MLflow Tracking Server
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Lab03_Tracking_Linear_Regression")
    
    np.random.seed(42)
    X = np.linspace(1, 100, 50).reshape(-1, 1)
    y = 3.2 * X.squeeze() + 15.0 + np.random.normal(0, 8, size=50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    with mlflow.start_run(run_name="central_tracked_lr_run") as run:
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # Log Parameters
        mlflow.log_params({
            "fit_intercept": True,
            "test_size": 0.25,
            "random_state": 42,
            "student_usn": "1MS25CI125",
            "server_uri": TRACKING_URI
        })
        
        # Log Metrics
        mlflow.log_metrics({
            "mae": float(mae),
            "r2_score": float(r2),
            "learned_coef": float(model.coef_[0]),
            "learned_intercept": float(model.intercept_)
        })
        
        # Log Tags
        mlflow.set_tags({
            "student_name": "Sujal Chopra",
            "lab": "Lab03",
            "entity": "LinearRegression"
        })
        
        print("=" * 60)
        print("Lab 03 - Server Tracked Linear Regression")
        print("=" * 60)
        print(f"Tracking Server URI : {TRACKING_URI}")
        print(f"Run ID              : {run.info.run_id}")
        print(f"MAE                 : {mae:.4f}")
        print(f"R² Score            : {r2:.4f}")
        print("Run logged to MLflow Server successfully.")

if __name__ == "__main__":
    main()

"""
Lab 10 — Model Evaluation across Experiments
Creates two distinct experiments to demonstrate how metric schemas and evaluation depth evolve:
1. 'Linear Regression Experiment' (basic: MAE only)
2. 'Linear Regression Metrics Experiment' (comprehensive: MAE, MSE, RMSE, R2)
Student: Sujal Chopra (1MS25CI125)
"""

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    
    # Dataset
    np.random.seed(42)
    X = np.linspace(5, 50, 50).reshape(-1, 1)
    y = 3.0 * X.squeeze() + 12.0 + np.random.normal(0, 4, size=50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    # -------------------------------------------------------------
    # Experiment 1: Linear Regression Experiment (Baseline: MAE only)
    # -------------------------------------------------------------
    EXP1 = "Linear Regression Experiment"
    mlflow.set_experiment(EXP1)
    with mlflow.start_run(run_name="baseline_mae_eval") as run1:
        mlflow.log_params({
            "model_type": "LinearRegression",
            "fit_intercept": True,
            "test_size": 0.3,
            "student_usn": "1MS25CI125"
        })
        mlflow.log_metric("test_mae", float(mae))
        mlflow.set_tag("student_name", "Sujal Chopra")
        print("=" * 65)
        print(f"Logged to Experiment 1: '{EXP1}'")
        print(f"Run ID: {run1.info.run_id}")
        print(f"Metrics Logged: test_mae = {mae:.4f}")
        
    # -------------------------------------------------------------
    # Experiment 2: Linear Regression Metrics Experiment (Multi-metric)
    # -------------------------------------------------------------
    EXP2 = "Linear Regression Metrics Experiment"
    mlflow.set_experiment(EXP2)
    with mlflow.start_run(run_name="comprehensive_metrics_eval") as run2:
        mlflow.log_params({
            "model_type": "LinearRegression",
            "fit_intercept": True,
            "test_size": 0.3,
            "student_usn": "1MS25CI125"
        })
        mlflow.log_metrics({
            "test_mae": float(mae),
            "test_mse": float(mse),
            "test_rmse": float(rmse),
            "test_r2": float(r2)
        })
        mlflow.set_tag("student_name", "Sujal Chopra")
        print("\n" + "=" * 65)
        print(f"Logged to Experiment 2: '{EXP2}'")
        print(f"Run ID: {run2.info.run_id}")
        print(f"Metrics Logged:")
        print(f"  test_mae  : {mae:.4f}")
        print(f"  test_mse  : {mse:.4f}")
        print(f"  test_rmse : {rmse:.4f}")
        print(f"  test_r2   : {r2:.4f}")
        print("=" * 65)

if __name__ == "__main__":
    main()

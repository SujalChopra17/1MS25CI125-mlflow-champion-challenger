"""
Lab 06 — MLflow Autologging vs Manual Logging
Compares automated tracking via mlflow.sklearn.autolog() against explicit manual tracking.
Student: Sujal Chopra (1MS25CI125)
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    EXP_NAME = "Lab06_Autolog_vs_Manual"
    mlflow.set_experiment(EXP_NAME)
    
    np.random.seed(42)
    X = np.linspace(1, 50, 60).reshape(-1, 1)
    y = 4.0 * X.squeeze() + 10.0 + np.random.normal(0, 5, size=60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print("=" * 70)
    print("Lab 06 — MLflow Autologging vs Explicit Manual Logging")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # Run 1: Autologging
    # -------------------------------------------------------------
    print("\n[+] Starting Run 1: Automated Tracking (mlflow.sklearn.autolog())")
    mlflow.sklearn.autolog(log_models=True)
    
    with mlflow.start_run(run_name="run_autologged") as run_auto:
        auto_run_id = run_auto.info.run_id
        model_auto = LinearRegression(fit_intercept=True)
        model_auto.fit(X_train, y_train)
        preds_auto = model_auto.predict(X_test)
        
        mlflow.set_tag("logging_mode", "automatic")
        mlflow.set_tag("student_usn", "1MS25CI125")
        mlflow.set_tag("student_name", "Sujal Chopra")
        print(f"    Autologged Run ID: {auto_run_id}")
        
    # Disable autologging before Run 2
    mlflow.sklearn.autolog(disable=True)
    
    # -------------------------------------------------------------
    # Run 2: Manual Logging
    # -------------------------------------------------------------
    print("\n[+] Starting Run 2: Explicit Manual Logging")
    with mlflow.start_run(run_name="run_manually_logged") as run_manual:
        manual_run_id = run_manual.info.run_id
        model_manual = LinearRegression(fit_intercept=True)
        model_manual.fit(X_train, y_train)
        preds_manual = model_manual.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds_manual)
        r2 = r2_score(y_test, preds_manual)
        
        # Explicit Parameters
        mlflow.log_params({
            "fit_intercept": True,
            "copy_X": True,
            "n_jobs": None,
            "positive": False,
            "test_size": 0.25,
            "random_state": 42
        })
        
        # Explicit Metrics
        mlflow.log_metrics({
            "test_mae": float(mae),
            "test_r2": float(r2),
            "slope": float(model_manual.coef_[0]),
            "intercept": float(model_manual.intercept_)
        })
        
        # Explicit Model Logging
        mlflow.sklearn.log_model(
            sk_model=model_manual,
            artifact_path="model",
            input_example=X_train[:2]
        )
        
        mlflow.set_tag("logging_mode", "manual")
        mlflow.set_tag("student_usn", "1MS25CI125")
        mlflow.set_tag("student_name", "Sujal Chopra")
        print(f"    Manual Run ID: {manual_run_id}")
        print(f"    Manually calculated test MAE: {mae:.4f}, R²: {r2:.4f}")
        
    print("\n" + "=" * 70)
    print("Both runs recorded in MLflow experiment: 'Lab06_Autolog_vs_Manual'")
    print("=" * 70)

if __name__ == "__main__":
    main()

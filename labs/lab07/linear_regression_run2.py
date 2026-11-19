"""
Lab 07 — Script 2: Linear Regression without Intercept
Student: Sujal Chopra (1MS25CI125)
"""

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    EXP_NAME = "Lab07_Shared_Tracking_Server"
    
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXP_NAME)
    
    np.random.seed(42)
    X = np.linspace(1, 20, 40).reshape(-1, 1)
    y = 2.5 * X.squeeze() + 5.0 + np.random.normal(0, 2, size=40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    with mlflow.start_run(run_name="script2_without_intercept") as run:
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_params({
            "script": "linear_regression_run2.py",
            "fit_intercept": False,
            "student_usn": "1MS25CI125"
        })
        
        mlflow.log_metrics({
            "mae": float(mae),
            "r2": float(r2),
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_)
        })
        
        mlflow.set_tag("student_name", "Sujal Chopra")
        
        print("=" * 60)
        print("Lab 07 - Script 2 (Without Intercept) Completed")
        print("=" * 60)
        print(f"Server URI : {TRACKING_URI}")
        print(f"Run ID     : {run.info.run_id}")
        print(f"Slope      : {model.coef_[0]:.4f}")
        print(f"Intercept  : {model.intercept_:.4f}")
        print(f"MAE        : {mae:.4f}")
        print(f"R²         : {r2:.4f}")

if __name__ == "__main__":
    main()

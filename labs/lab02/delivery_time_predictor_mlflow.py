"""
Lab 02 — Exercise 2: Delivery Time Predictor with MLflow
Tracking delivery time linear regression training with MLflow params, metrics, and artifacts.
Student: Sujal Chopra (1MS25CI125)
"""

import os
import mlflow
import pandas as pd
import numpy as np

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Lab02_Delivery_Time_Predictor")
    
    # Official Delivery dataset
    data = {
        "order": [1, 2, 3, 4, 5],
        "distance_x_km": [1, 2, 3, 4, 5],
        "actual_time_y_min": [16, 19, 27, 29, 34]
    }
    df = pd.DataFrame(data)
    X = df[["distance_x_km"]]
    y = df["actual_time_y_min"]
    
    test_size = 0.2
    random_state = 42
    
    with mlflow.start_run(run_name="delivery_linear_regression") as run:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        # Log Parameters
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_family", "LinearRegression")
        mlflow.log_param("student_usn", "1MS25CI125")
        
        # Log Metrics
        mlflow.log_metric("n_train_samples", len(X_train))
        mlflow.log_metric("n_test_samples", len(X_test))
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("model_slope", float(model.coef_[0]))
        mlflow.log_metric("model_intercept", float(model.intercept_))
        
        # Log Tags
        mlflow.set_tag("student_name", "Sujal Chopra")
        mlflow.set_tag("task", "delivery_time_prediction")
        
        # Log artifact
        pred_df = pd.DataFrame({
            "order": df.loc[X_test.index, "order"],
            "distance_km": X_test["distance_x_km"].values,
            "actual_time_min": y_test.values,
            "predicted_time_min": preds,
            "error_min": np.abs(y_test.values - preds)
        })
        os.makedirs("outputs", exist_ok=True)
        pred_csv = "outputs/delivery_predictions.csv"
        pred_df.to_csv(pred_csv, index=False)
        mlflow.log_artifact(pred_csv)
        
        print("=" * 60)
        print("Lab 02 - MLflow Tracked Delivery Predictor")
        print("=" * 60)
        print(f"Run ID: {run.info.run_id}")
        print(f"MAE: {mae:.4f}")
        print(f"Predictions logged to: {pred_csv}")
        print(pred_df.to_string(index=False))

if __name__ == "__main__":
    main()

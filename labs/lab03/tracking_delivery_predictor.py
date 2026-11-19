"""
Lab 03 — Exercise 2: Server-Tracked Delivery Time Predictor
Student: Sujal Chopra (1MS25CI125)
"""

import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Lab03_Delivery_Predictor_Server")
    
    # Official Delivery dataset from Lab Book
    data = {
        "order": [1, 2, 3, 4, 5],
        "distance_x_km": [1, 2, 3, 4, 5],
        "actual_time_y_min": [16, 19, 27, 29, 34]
    }
    df = pd.DataFrame(data)
    X = df[["distance_x_km"]]
    y = df["actual_time_y_min"]
    
    with mlflow.start_run(run_name="delivery_server_run") as run:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        mlflow.log_params({
            "test_size": 0.2,
            "random_state": 42,
            "student_usn": "1MS25CI125",
            "model_type": "LinearRegression"
        })
        
        mlflow.log_metrics({
            "mae": float(mae),
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_)
        })
        
        mlflow.set_tags({
            "student_name": "Sujal Chopra",
            "server": "localhost:5000"
        })
        
        # Save and log predictions table
        os.makedirs("outputs", exist_ok=True)
        pred_path = "outputs/delivery_server_preds.csv"
        res_df = pd.DataFrame({
            "distance_km": X_test["distance_x_km"].values,
            "actual_time": y_test.values,
            "pred_time": preds,
            "abs_error": np.abs(y_test.values - preds)
        })
        res_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path)
        
        print("=" * 60)
        print("Lab 03 - Delivery Predictor Run on Server")
        print("=" * 60)
        print(f"Server URI : {TRACKING_URI}")
        print(f"Run ID     : {run.info.run_id}")
        print(f"MAE        : {mae:.4f}")
        print("Entities recorded in MLflow UI.")

if __name__ == "__main__":
    main()

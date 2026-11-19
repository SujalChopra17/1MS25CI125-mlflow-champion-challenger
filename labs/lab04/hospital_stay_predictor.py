"""
Lab 04 — MLflow Logging Functions: Hospital Stay Predictor
Uses exact 20-patient dataset from the Lab Book to predict hospital stay in days (Y)
from days with symptoms before admission (X), logging metrics and a detailed prediction error table.
Student: Sujal Chopra (1MS25CI125)
"""

import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    TRACKING_URI = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Lab04_Hospital_Stay_Predictor")
    
    # Exact 20 Patients data from Lab Book
    patients = [
        ("P001", 1, 2),  ("P002", 2, 3),  ("P003", 3, 3),  ("P004", 4, 5),  ("P005", 5, 5),
        ("P006", 6, 7),  ("P007", 7, 7),  ("P008", 8, 8),  ("P009", 9, 10), ("P010", 10, 9),
        ("P011", 2, 2),  ("P012", 4, 4),  ("P013", 6, 6),  ("P014", 8, 9),  ("P015", 3, 4),
        ("P016", 5, 6),  ("P017", 7, 6),  ("P018", 9, 9),  ("P019", 1, 3),  ("P020", 10, 11)
    ]
    df = pd.DataFrame(patients, columns=["patient_id", "symptom_days_x", "hospital_stay_y"])
    
    test_size = 0.25
    random_state = 42
    
    X = df[["symptom_days_x"]]
    y = df["hospital_stay_y"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    with mlflow.start_run(run_name="hospital_stay_regression") as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # 1. Log Observations count as parameters / metrics
        mlflow.log_params({
            "model_type": "LinearRegression",
            "test_size": test_size,
            "random_state": random_state,
            "total_patients": len(df),
            "student_usn": "1MS25CI125"
        })
        
        mlflow.log_metrics({
            "n_training_observations": len(X_train),
            "n_testing_observations": len(X_test),
            "mae": float(mae),
            "mse": float(mse),
            "r2_score": float(r2),
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_)
        })
        
        # 2. Table of actual, predicted hospital days with error for each test row
        test_patient_ids = df.loc[X_test.index, "patient_id"].values
        results_df = pd.DataFrame({
            "patient_id": test_patient_ids,
            "symptom_days_X": X_test["symptom_days_x"].values,
            "actual_days_Y": y_test.values,
            "predicted_days": np.round(preds, 2),
            "absolute_error": np.round(np.abs(y_test.values - preds), 2)
        })
        
        os.makedirs("outputs", exist_ok=True)
        table_path = "outputs/hospital_stay_predictions_table.csv"
        results_df.to_csv(table_path, index=False)
        mlflow.log_artifact(table_path)
        
        # Log table directly to MLflow if supported
        try:
            mlflow.log_table(data=results_df, artifact_file="hospital_stay_table.json")
        except Exception:
            pass
        
        mlflow.set_tags({
            "student_name": "Sujal Chopra",
            "clinical_study": "hospital_length_of_stay"
        })
        
        print("=" * 65)
        print("Lab 04 — Hospital Stay Predictor (Logging Functions)")
        print("=" * 65)
        print(f"Number of training observations : {len(X_train)}")
        print(f"Number of testing observations  : {len(X_test)}")
        print(f"Model Learned Equation          : Stay = {model.coef_[0]:.4f} * X + {model.intercept_:.4f}")
        print(f"Mean Absolute Error (MAE)       : {mae:.4f} days")
        print(f"Mean Squared Error (MSE)        : {mse:.4f}")
        print(f"R² Score                        : {r2:.4f}")
        print("\nTest Prediction & Error Table:")
        print(results_df.to_string(index=False))
        print("=" * 65)

if __name__ == "__main__":
    main()
